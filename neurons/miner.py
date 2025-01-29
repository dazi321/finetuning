async def main(config: bt.config):
    # Create bittensor objects.
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

    # If running online, make sure the miner is registered, has a hugging face access token, and has provided a repo id.
    my_uid = None
    if not config.offline:
        my_uid = metagraph_utils.assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()

    # Data comes from Subnet 1's wandb project. Make sure we're logged in
    wandb_utils.login()

    # Create a unique run id for this run.
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = ft.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)

    use_wandb = False
    if not config.offline:
        if config.wandb_project is None or config.wandb_entity is None:
            logging.warning(
                "Wandb project or entity not specified. This run will not be logged to wandb"
            )
        else:
            use_wandb = True

    model_constraints = constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(
        config.competition_id, None
    )

    if not model_constraints:
        raise RuntimeError(f"No competition found for {config.competition_id}")
    kwargs = model_constraints.kwargs.copy()
    kwargs["torch_dtype"] = (
        torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    )

    # Init model.
    tokenizer = ft.model.load_tokenizer(model_constraints, cache_dir=config.model_dir)
    model = await load_starting_model(config, metagraph, chain_metadata_store, kwargs)
    model = model.train()
    model = model.to(config.device)

    logging.info(f"Saving model to path: {model_dir}.")
    ft.mining.save(model, model_dir)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    wandb_run = None

    # If using wandb, start a new run.
    if use_wandb:
        token = os.getenv("WANDB_API_KEY")
        if not token:
            raise ValueError(
                "To use Wandb, you must set WANDB_API_KEY in your .env file"
            )

        wandb.login(key=token)

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
    else:
        logging.warning(
            "Not posting run to wandb. Either --offline is specified or the wandb settings are missing."
        )

    # Start the training loop
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    best_avg_deviation = math.inf
    accumulation_steps = config.accumulation_steps

    try:
        while epoch_step < config.num_epochs or config.num_epochs == -1:
            # Your training logic here
            optimizer.zero_grad()
            
            # Dummy data (replace with your actual data and forward pass)
            input_data = torch.randn(config.batch_size, config.input_dim).to(config.device)
            target_data = torch.randn(config.batch_size, config.output_dim).to(config.device)

            # Forward pass
            outputs = model(input_data)
            loss = compute_loss(outputs, target_data)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            global_step += 1
            n_acc_steps += 1

            # Log metrics
            if use_wandb and global_step % config.log_interval == 0:
                wandb.log({
                    "loss": loss.item(),
                    "global_step": global_step,
                    "epoch_step": epoch_step,
                })

            # Save the model periodically
            if global_step % config.save_interval == 0:
                logging.info(f"Saving model at global step: {global_step}")
                ft.mining.save(model, model_dir)

            epoch_step += 1
            logging.info(f"Epoch step: {epoch_step}, Global step: {global_step}")

        logging.info("Finished training")

        # Push the model to hugging face if criteria are met
        if not config.offline:
            if best_avg_deviation < config.avg_loss_upload_threshold:
                logging.info(
                    f"Trained model had a best_avg_deviation of {best_avg_deviation} which is below the threshold of {config.avg_loss_upload_threshold}. Uploading to hugging face. "
                )

                # First, reload the best model from the training run.
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
            else:
                logging.info(
                    f"This training run achieved a best_avg_deviation={best_avg_deviation}, which did not meet the upload threshold. Not uploading to hugging face."
                )
        else:
            logging.info(
                "Not uploading to hugging face because --offline was specified."
            )

    finally:
        # Important step.
        if wandb_run:
            wandb_run.finish()

if __name__ == "__main__":
    # Parse and print configuration
    config = neuron_config.miner_config()

    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE_BY_BLOCK)
    else:
        print(config)
        asyncio.run(main(config))
