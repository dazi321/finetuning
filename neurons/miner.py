
import asyncio
import datetime as dt
import math
import os
import typing

import bittensor as bt
import torch
import wandb
from dotenv import load_dotenv
from taoverse.metagraph import utils as metagraph_utils
from taoverse.model.data import Model
from taoverse.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from taoverse.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from taoverse.model.storage.model_metadata_store import ModelMetadataStore
from taoverse.utilities import utils as taoverse_utils
from taoverse.utilities import wandb as wandb_utils

import constants
import finetune as ft
from neurons import config as neuron_config
import taoverse.utilities.logging as logging

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"

async def load_starting_model(
    config: bt.config,
    metagraph: bt.metagraph,
    metadata_store: ModelMetadataStore,
    kwargs: typing.Dict[str, typing.Any],
) -> Model:
    if config.load_best:
        model = await ft.mining.load_best_model(
            download_dir=config.model_dir,
            competition_id=config.competition_id,
            metagraph=metagraph,
            metadata_store=metadata_store,
        )
        logging.info(f"Training with best model from competition: {config.competition_id}. Model={str(model)}")
        return model

    if config.load_uid is not None:
        model = await ft.mining.load_remote_model(
            config.load_uid,
            config.model_dir,
            metagraph=metagraph,
            metadata_store=metadata_store,
        )
        logging.info(f"Training with model from uid: {config.load_uid}. Model={str(model)}")
        return model

    if config.load_model_dir:
        model = ft.mining.load_local_model(
            config.load_model_dir, config.competition_id, kwargs
        )
        logging.info(f"Training with model from disk. Model={str(model)}")
        return model

    raise RuntimeError("No starting model specified, pass either --load_best, --load_uid, or --load_model_dir")

async def main(config: bt.config):
    bt.logging.set_warning()
    taoverse_utils.logging.reinitialize()
    taoverse_utils.configure_logging(config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    chain_metadata_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=config.netuid,
        wallet=wallet,
    )

    my_uid = None
    if not config.offline:
        my_uid = metagraph_utils.assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()

    wandb_utils.login()

    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = ft.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)

    use_wandb = not config.offline and config.wandb_project and config.wandb_entity

    model_constraints = constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(
        config.competition_id, None
    )
    if not model_constraints:
        raise RuntimeError(f"No competition found for {config.competition_id}")

    kwargs = model_constraints.kwargs.copy()
    kwargs["torch_dtype"] = (
        torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    )

    tokenizer = ft.model.load_tokenizer(
    model_constraints, cache_dir=config.model_dir, token=os.getenv("HF_ACCESS_TOKEN")
)

    model = await load_starting_model(config, metagraph, chain_metadata_store, kwargs)
    model = model.train().to(config.device)

    logging.info(f"Saving model to path: {model_dir}.")
    ft.mining.save(model, model_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            name=run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={
                "uid": my_uid,
                "hotkey": wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__version__,
                "type": "miner",
            },
            allow_val_change=True,
        )

    best_avg_deviation = math.inf
    for epoch in range(config.num_epochs if config.num_epochs > 0 else float('inf')):
        epoch_loss = 0.0
        for step, batch in enumerate(ft.data_loader(config.batch_size)):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if use_wandb:
                wandb.log({"loss": loss.item()})

        avg_deviation = epoch_loss / (step + 1)
        logging.info(f"Epoch {epoch}, Avg Loss: {avg_deviation}")
        
        if avg_deviation < best_avg_deviation:
            best_avg_deviation = avg_deviation
            logging.info(f"New best avg deviation: {best_avg_deviation}")
            logging.info(f"Saving model to path: {model_dir}.")
            ft.mining.save(model, model_dir)

    if not config.offline and best_avg_deviation < config.avg_loss_upload_threshold:
        model_to_upload = ft.mining.load_local_model(
            model_dir, config.competition_id, model_constraints.kwargs
        )
        await ft.mining.push(
            model_to_upload,
            config.hf_repo_id,
            config.competition_id,
            wallet,
            update_repo_visibility=config.update_repo_visibility,
            metadata_store=chain_metadata_store,
        )

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    config = neuron_config.miner_config()
    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE_BY_BLOCK)
    else:
        print(config)
        asyncio.run(main(config))
