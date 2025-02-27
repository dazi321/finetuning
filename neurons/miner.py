import asyncio
import datetime as dt
import math
import os
import typing

import bittensor as bt
import torch
import torch.nn.functional as F
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

load_dotenv()  # Load environment variables

torch.backends.cudnn.benchmark = True  # Enable CUDA optimization

async def train_model(config, model, optimizer, dataloader, device, wandb_run):
    """Training logic for the miner."""
    model.train()
    best_avg_loss = math.inf
    
    for epoch in range(config.num_epochs if config.num_epochs != -1 else float('inf')):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log to WandB
            if wandb_run:
                wandb_run.log({"batch_loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch + 1}: Avg Loss = {avg_loss}")
        
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            logging.info(f"New best loss: {best_avg_loss}, saving model...")
            ft.mining.save(model, config.model_dir)
    
    return best_avg_loss

async def main(config: bt.config):
    # Authentication and initialization
    bt.logging.set_warning()
    taoverse_utils.logging.reinitialize()
    taoverse_utils.configure_logging(config)
    
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    chain_metadata_store = ChainModelMetadataStore(subtensor=subtensor, subnet_uid=config.netuid, wallet=wallet)
    
    my_uid = None
    if not config.offline:
        my_uid = metagraph_utils.assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()
    
    wandb_utils.login()
    
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = ft.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)
    
    use_wandb = not config.offline and config.wandb_project and config.wandb_entity
    
    kwargs = constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID[config.competition_id].kwargs.copy()
    kwargs["torch_dtype"] = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    
    tokenizer = ft.model.load_tokenizer(constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID[config.competition_id], cache_dir=config.model_dir)
    model = await load_starting_model(config, metagraph, chain_metadata_store, kwargs)
    model.to(config.device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            name=run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={"uid": my_uid, "hotkey": wallet.hotkey.ss58_address, "run_name": run_id, "version": constants.__version__, "type": "miner"},
            allow_val_change=True,
        )
    
    dataloader = ft.mining.get_dataloader(config)
    best_loss = await train_model(config, model, optimizer, dataloader, config.device, wandb_run)
    
    if best_loss < config.avg_loss_upload_threshold and not config.offline:
        logging.info("Uploading best model to Hugging Face...")
        model_to_upload = ft.mining.load_local_model(model_dir, config.competition_id, kwargs)
        await ft.mining.push(model_to_upload, config.hf_repo_id, config.competition_id, wallet, update_repo_visibility=config.update_repo_visibility, metadata_store=chain_metadata_store)
    
    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    config = neuron_config.miner_config()
    asyncio.run(main(config))
