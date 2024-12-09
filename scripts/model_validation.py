"""This script evaluates a local model using the same process as a Validator.

It can be used to estimate the performance of a model before submitting it."""

import argparse
import datetime as dt
import random
import sys
from typing import List

import bittensor as bt
import nltk
from taoverse.metagraph import utils as metagraph_utils
from taoverse.model.competition import utils as competition_utils
from taoverse.model.eval.task import EvalTask
from taoverse.model.model_updater import ModelUpdater
from taoverse.utilities.enum_action import IntEnumAction

import constants
import finetune as ft
from competitions.data import CompetitionId
from finetune.datasets.factory import DatasetLoaderFactory
from finetune.datasets.ids import DatasetId
from finetune.datasets.subnet.prompting_subset_loader import PromptingSubsetLoader
from finetune.eval.sample import EvalSample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="Local path to your model", required=True
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed to use while loading data. If 0 then randomize.",
    )
    parser.add_argument(
        "--competition_id",
        type=CompetitionId,
        default=CompetitionId.B7_MULTI_CHOICE.value,
        action=IntEnumAction,
        help="competition to mine for (use --list-competitions to get all competitions)",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )
    parser.add_argument(
        "--comp_block",
        type=int,
        default=9999999999,
        help="Block to lookup competition id from.",
    )
    args = parser.parse_args()
    if args.list_competitions:
        print(
            competition_utils.get_competition_schedule_for_block(
                args.comp_block, constants.COMPETITION_SCHEDULE_BY_BLOCK
            )
        )
        return

    competition = competition_utils.get_competition_for_block(
        args.competition_id,
        args.comp_block,
        constants.COMPETITION_SCHEDULE_BY_BLOCK,
    )

    if not competition:
        print(f"Competition {args.competition_id} not found.")
        return

    kwargs = competition.constraints.kwargs.copy()
    kwargs["use_cache"] = True

    print(f"Loading tokenizer and model from {args.model_path}")
    model = ft.mining.load_local_model(args.model_path, kwargs)

    if competition.constraints.tokenizer:
        model.tokenizer = ft.model.load_tokenizer(competition.constraints)

    if not ModelUpdater.verify_model_satisfies_parameters(
        model, competition.constraints
    ):
        print("Model does not satisfy competition parameters!!!")
        return

    seed = args.random_seed if args.random_seed else random.randint(0, sys.maxsize)

    print("Loading evaluation tasks")
    eval_tasks: List[EvalTask] = []
    samples: List[List[EvalSample]] = []

    # Load data based on the competition.
    metagraph = bt.metagraph(constants.PROMPTING_SUBNET_UID)
    vali_uids = metagraph_utils.get_high_stake_validators(
        metagraph, constants.SAMPLE_VALI_MIN_STAKE
    )
    vali_hotkeys = set([metagraph.hotkeys[uid] for uid in vali_uids])

    for eval_task in competition.eval_tasks:
        if eval_task.dataset_id == DatasetId.SYNTHETIC_MMLU:
            data_loader = PromptingSubsetLoader(
                random_seed=seed,
                oldest_sample_timestamp=dt.datetime.now(dt.timezone.utc)
                - dt.timedelta(hours=6),
                validator_hotkeys=vali_hotkeys,
            )
        else:
            data_loader = DatasetLoaderFactory.get_loader(
                dataset_id=eval_task.dataset_id,
                dataset_kwargs=eval_task.dataset_kwargs,
                seed=seed,
                validator_hotkeys=vali_hotkeys,
            )

        if data_loader:
            eval_tasks.append(eval_task)
            print(f"Loaded {len(data_loader)} samples for task {eval_task.name}")
            samples.append(
                data_loader.tokenize(
                    model.tokenizer, competition.constraints.sequence_length
                )
            )

    print(f"Scoring model on tasks {eval_tasks}")
    # Run each computation in a subprocess so that the GPU is reset between each model.
    score, score_details = ft.validation.score_model(
        model,
        eval_tasks,
        samples,
        competition,
        args.device,
    )

    print(f"Computed score: {score}. Details: {score_details}")


if __name__ == "__main__":
    # Make sure we can download the needed ntlk modules
    nltk_modules = {
        "words",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    }
    for module in nltk_modules:
        nltk.download(module, raise_on_error=True)

    main()
