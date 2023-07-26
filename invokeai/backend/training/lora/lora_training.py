import json
import logging
import math
import os
import random
import shutil
import time

import datasets
import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import invokeai.backend.training.lora.networks.lora as kohya_lora_module
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_manager_service import ModelManagerService
from invokeai.backend.model_management.models.base import SubModelType
from invokeai.backend.training.lora.lib.original_unet import (
    UNet2DConditionModel as KohyaUNet2DConditionModel,
)
from invokeai.backend.training.lora.lora_training_config import (
    LoraTrainingConfig,
)
from invokeai.backend.training.lora.networks.lora import LoRANetwork


def _initialize_accelerator(
    out_dir: str, train_config: LoraTrainingConfig
) -> Accelerator:
    """Configure Hugging Face accelerate and return an Accelerator.

    Args:
        out_dir (str): The output directory where results will be written.
        train_config (LoraTrainingConfig): LoRA training configuration.

    Returns:
        Accelerator
    """
    accelerator_project_config = ProjectConfiguration(
        project_dir=out_dir,
        logging_dir=os.path.join(out_dir, "logs"),
    )
    return Accelerator(
        project_config=accelerator_project_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=train_config.report_to,
    )


def _initialize_logging(accelerator: Accelerator) -> MultiProcessAdapter:
    """Configure logging.

    Returns an accelerate logger with multi-process logging support. Logging is
    configured to be more verbose on the main process. Non-main processes only
    log at error level for Hugging Face libraries (datasets, transformers,
    diffusers).

    Args:
        accelerator (Accelerator): _description_

    Returns:
        MultiProcessAdapter: _description_
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # Only log errors from non-main processes.
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return get_logger(__name__)


def _get_weight_type(accelerator: Accelerator):
    """Extract torch.dtype from Accelerator config.

    Args:
        accelerator (Accelerator): The Hugging Face Accelerator.

    Raises:
        NotImplementedError: If the accelerator's mixed_precision configuration
        is not recognized.

    Returns:
        torch.dtype: The weight type inferred from the accelerator
        mixed_precision configuration.
    """
    weight_dtype: torch.dtype = torch.float32
    if (
        accelerator.mixed_precision is None
        or accelerator.mixed_precision == "no"
    ):
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        # TODO(ryand): Add support for more precision types (specifically bf16)
        # and test.
        raise NotImplementedError(
            f"mixed_precision mode '{accelerator.mixed_precision}' is not yet"
            " supported."
        )

    return weight_dtype


def _load_models(
    accelerator: Accelerator,
    app_config: InvokeAIAppConfig,
    train_config: LoraTrainingConfig,
    logger: logging.Logger,
) -> tuple[
    CLIPTokenizer,
    DDPMScheduler,
    CLIPTextModel,
    AutoencoderKL,
    KohyaUNet2DConditionModel,
]:
    """Load all models required for training from disk, transfer them to the
    target training device and cast their weight dtypes.

    Args:
        app_config (InvokeAIAppConfig): The app config.
        train_config (LoraTrainingConfig): The LoRA training run config.
        logger (logging.Logger): A logger.

    Returns:
        tuple[
            CLIPTokenizer,
            DDPMScheduler,
            CLIPTextModel,
            AutoencoderKL,
            KohyaUNet2DConditionModel,
        ]: A tuple of loaded models.
    """
    model_manager = ModelManagerService(app_config, logger)

    known_models = model_manager.model_names()

    model_name = train_config.model.split("/")[-1]

    # Find the first known model that matches model_name. Raise an exception if
    # there is no match.
    model_meta = next(
        (mm for mm in known_models if mm[0].endswith(model_name)), None
    )
    assert model_meta is not None, f"Unknown model: {train_config.model}"

    # Validate that the model is a diffusers model.
    model_info = model_manager.model_info(*model_meta)
    model_format = model_info["model_format"]
    assert model_format == "diffusers", (
        "LoRA training only supports models in the 'diffusers' format."
        f" '{train_config.model}' is in the '{model_format}' format. "
    )

    # Get sub-model info.
    tokenizer_info = model_manager.get_model(
        *model_meta, submodel=SubModelType.Tokenizer
    )
    noise_scheduler_info = model_manager.get_model(
        *model_meta, submodel=SubModelType.Scheduler
    )
    text_encoder_info = model_manager.get_model(
        *model_meta, submodel=SubModelType.TextEncoder
    )
    vae_info = model_manager.get_model(*model_meta, submodel=SubModelType.Vae)
    unet_info = model_manager.get_model(*model_meta, submodel=SubModelType.UNet)

    # Load all models.
    pipeline_args = dict(local_files_only=True)
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        tokenizer_info.location, subfolder="tokenizer", **pipeline_args
    )
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
        noise_scheduler_info.location, subfolder="scheduler", **pipeline_args
    )
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        text_encoder_info.location, subfolder="text_encoder", **pipeline_args
    )
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        vae_info.location, subfolder="vae", **pipeline_args
    )
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        unet_info.location, subfolder="unet", **pipeline_args
    )

    # Convert unet to "original unet" to replicate the behavior of kohya_ss.
    # TODO(ryand): Eliminate the need for this step and just work directly with
    # diffusers models.
    original_unet = KohyaUNet2DConditionModel(
        unet.config.sample_size,
        unet.config.attention_head_dim,
        unet.config.cross_attention_dim,
        unet.config.use_linear_projection,
        unet.config.upcast_attention,
    )
    original_unet.load_state_dict(unet.state_dict())
    unet: KohyaUNet2DConditionModel = original_unet
    logger.info(
        "Converted UNet2DConditionModel to kohya_ss 'original'"
        " UNet2DConditionModel."
    )

    # Disable gradient calculation for model weights to save memory.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Put models in 'eval' mode.
    text_encoder.eval()
    vae.eval()
    unet.eval()

    weight_dtype = _get_weight_type(accelerator)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    return tokenizer, noise_scheduler, text_encoder, vae, unet


def _initialize_optimizer(
    train_config: LoraTrainingConfig, trainable_params: list
) -> torch.optim.Optimizer:
    """Initialize an optimizer based on the train_config."""
    # TODO(ryand): Add support for 8-bit Adam optimizer.
    return torch.optim.AdamW(
        trainable_params,
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        weight_decay=train_config.adam_weight_decay,
        eps=train_config.adam_epsilon,
    )


def _initialize_dataset(
    train_config: LoraTrainingConfig,
    accelerator: Accelerator,
    tokenizer: CLIPTokenizer,
) -> torch.utils.data.DataLoader:
    # In distributed training, the load_dataset function guarantees that only
    # one local process will download the dataset.
    if train_config.dataset_name is not None:
        # Download the dataset from the Hugging Face hub.
        dataset = datasets.load_dataset(
            train_config.dataset_name,
            train_config.dataset_config_name,
            cache_dir=train_config.hf_cache_dir,
        )
    elif train_config.dataset_dir is not None:
        data_files = {}
        data_files["train"] = os.path.join(train_config.dataset_dir, "**")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=train_config.hf_cache_dir,
        )
    else:
        raise ValueError(
            "At least one of 'dataset_name' or 'dataset_dir' must be set."
        )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # Get the column names for input/target.
    if train_config.dataset_image_column not in column_names:
        raise ValueError(
            f"The dataset_image_column='{train_config.dataset_image_column}' is"
            f" not in the set of dataset column names: '{column_names}'."
        )
    if train_config.dataset_caption_column not in column_names:
        raise ValueError(
            f"The dataset_caption_column='{train_config.dataset_caption_column}'"
            f" is not in the set of dataset column names: '{column_names}'."
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[train_config.dataset_caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(
                    random.choice(caption) if is_train else caption[0]
                )
            else:
                raise ValueError(
                    f"Caption column `{train_config.dataset_caption_column}`"
                    " should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                train_config.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            (
                transforms.CenterCrop(train_config.resolution)
                if train_config.center_crop
                else transforms.RandomCrop(train_config.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if train_config.random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [
            image.convert("RGB")
            for image in examples[train_config.dataset_image_column]
        ]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack(
            [example["pixel_values"] for example in examples]
        )
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_config.train_batch_size,
        num_workers=train_config.dataloader_num_workers,
    )

    return train_dataloader


def _save_checkpoint(
    idx: int,
    prefix: str,
    out_dir: str,
    network: LoRANetwork,
    save_dtype: torch.dtype,
    train_config: LoraTrainingConfig,
    logger: logging.Logger,
):
    """Save a checkpoint. Old checkpoints are deleted if necessary to respect
    the train_config.max_checkpoints config.

    Args:
        idx (int): The checkpoint index (typically step count or epoch).
        prefix (str): The checkpoint naming prefix. Usually 'epoch' or 'step'.
        accelerator (Accelerator): Accelerator whose state will be saved.
        train_config (LoraTrainingConfig): Training configuration.
        logger (logging.Logger): Logger.
    """
    full_prefix = f"checkpoint_{prefix}-"

    # Before saving a checkpoint, check if this save would put us over the
    # max_checkpoints limit.
    if train_config.max_checkpoints is not None:
        checkpoints = os.listdir(out_dir)
        checkpoints = [d for d in checkpoints if d.startswith(full_prefix)]
        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(os.path.splitext(x)[0].split("-")[-1]),
        )

        if len(checkpoints) >= train_config.max_checkpoints:
            num_to_remove = len(checkpoints) - train_config.max_checkpoints + 1
            checkpoints_to_remove = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already"
                " exist. Removing"
                f" {len(checkpoints_to_remove)} checkpoints."
            )
            logger.info(f"Removing checkpoints: {checkpoints_to_remove}")

            for checkpoint_to_remove in checkpoints_to_remove:
                checkpoint_to_remove = os.path.join(
                    out_dir, checkpoint_to_remove
                )
                if os.path.isfile(checkpoint_to_remove):
                    # Delete checkpoint file.
                    os.remove(checkpoint_to_remove)
                else:
                    # Delete checkpoint directory.
                    shutil.rmtree(checkpoint_to_remove)

    save_path = os.path.join(out_dir, f"{full_prefix}{idx:0>8}")
    network.save_weights(save_path, save_dtype, None)
    # accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def _generate_validation_images(
    epoch: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: KohyaUNet2DConditionModel,
    train_config: LoraTrainingConfig,
    logger: logging.Logger,
):
    logger.info("Generating validation images.")

    # HACK(ryand): The KohyaUNet2DConditionModel model is based on an old
    # version of the diffusers.UNet2DConditionModel. We monkeypatch its `config`
    # fields to give it the entries expected by `StableDiffusionPipeline`.
    unet.config.in_channels = unet.in_channels
    unet.config.sample_size = unet.sample_size

    # Create pipeline.
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        # TODO(ryand): Add safety checker support.
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference.
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(train_config.validation_prompts):
            generator = torch.Generator(device=accelerator.device)
            if train_config.seed is not None:
                generator = generator.manual_seed(train_config.seed)

            images = []
            for _ in range(train_config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    images.append(
                        pipeline(
                            prompt,
                            num_inference_steps=30,
                            generator=generator,
                        ).images[0]
                    )

            # Save images to disk.
            validation_dir = os.path.join(
                out_dir,
                "validation",
                f"epoch_{epoch:0>8}",
                f"prompt_{prompt_idx:0>4}",
            )
            os.makedirs(validation_dir)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(validation_dir, f"{image_idx:0>4}.jpg"))

            # Log images to trackers. Currently, only tensorboard is supported.
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        f"validation (prompt {prompt_idx})",
                        np_images,
                        epoch,
                        dataformats="NHWC",
                    )

    del pipeline
    torch.cuda.empty_cache()


def run_lora_training(
    app_config: InvokeAIAppConfig, train_config: LoraTrainingConfig
):
    # Create a timestamped directory for all outputs.
    out_dir = os.path.join(train_config.base_output_dir, f"{time.time()}")
    os.makedirs(out_dir)

    accelerator = _initialize_accelerator(out_dir, train_config)
    logger = _initialize_logging(accelerator)

    # Set the accelerate seed.
    if train_config.seed is not None:
        set_seed(train_config.seed)

    # Log the accelerator configuration from every process to help with
    # debugging.
    logger.info(accelerator.state, main_process_only=False)

    logger.info("Starting LoRA Training.")
    logger.info(
        f"Configuration:\n{json.dumps(train_config.dict(), indent=2, default=str)}"
    )
    logger.info(f"Output dir: '{out_dir}'")

    # Write the configuration to disk.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(train_config.dict(), f, indent=2, default=str)

    weight_dtype = _get_weight_type(accelerator)

    tokenizer, noise_scheduler, text_encoder, vae, unet = _load_models(
        accelerator, app_config, train_config, logger
    )

    if train_config.xformers:
        import xformers

        unet.set_use_memory_efficient_attention(True, False)
        vae.enable_xformers_memory_efficient_attention()

    # Initialize LoRA network.
    lora_network = kohya_lora_module.create_network(
        multiplier=1.0,
        network_dim=None,
        network_alpha=None,
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
        neuron_dropout=None,
    )
    lora_network.apply_to(
        text_encoder, unet, apply_text_encoder=True, apply_unet=True
    )

    if train_config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        lora_network.enable_gradient_checkpointing()

        # At load time, the VAE, UNet, and text encoder are put into 'eval'
        # mode. According to kohya_ss, networks must be in 'train' mode to
        # enable gradient checkpointing.
        # TODO(ryand): Test that this is true, and test if it has other
        # implications (e.g. dropout, batch norm, etc.).
        unet.train()
        text_encoder.train()

        # This fix is from:
        # https://github.com/kohya-ss/sd-scripts/commit/e6a8c9d269b4952a6944dfe0e78a1f89bd036971
        # Without it, training fails.
        # TODO(ryand): Investigate and document more clearly why this is
        # necessary.
        text_encoder.text_model.embeddings.requires_grad_(True)

    trainable_params = lora_network.prepare_optimizer_params(
        text_encoder_lr=None,
        unet_lr=None,
        default_lr=train_config.learning_rate,
    )

    optimizer = _initialize_optimizer(train_config, trainable_params)

    data_loader = _initialize_dataset(train_config, accelerator, tokenizer)

    # TODO(ryand): Revisit and more clearly document the definition of 'steps'.
    # Consider interactions with batch_size, gradient_accumulation_steps, and
    # number of training processes.
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
        train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps
        * train_config.gradient_accumulation_steps,
        num_training_steps=train_config.max_train_steps
        * train_config.gradient_accumulation_steps,
    )

    prepared_result: tuple[
        KohyaUNet2DConditionModel,
        CLIPTextModel,
        LoRANetwork,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(
        unet,
        text_encoder,
        lora_network,
        optimizer,
        data_loader,
        lr_scheduler,
    )
    (
        unet,
        text_encoder,
        lora_network,
        optimizer,
        dataloader,
        lr_scheduler,
    ) = prepared_result

    # Calculate number of epochs and total training steps.
    # Note: A "step" represents a single optimizer weight update operation (i.e.
    # takes into account gradient accumulation steps).
    num_steps_per_epoch = math.ceil(
        len(dataloader) / train_config.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        train_config.max_train_steps / num_steps_per_epoch
    )

    lora_network.prepare_grad_etc(text_encoder, unet)

    # Initialize the trackers we use, and store the training configuration.
    if accelerator.is_main_process:
        accelerator.init_trackers("lora_training")

    # Train!
    total_batch_size = (
        train_config.train_batch_size
        * accelerator.num_processes
        * train_config.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_loader)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        "  Instantaneous batch size per device ="
        f" {train_config.train_batch_size}"
    )
    logger.info(
        "  Gradient accumulation steps ="
        f" {train_config.gradient_accumulation_steps}"
    )
    logger.info(f"  Parallel processes = {accelerator.num_processes}")
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) ="
        f" {total_batch_size}"
    )
    logger.info(f"  Total optimization steps = {train_config.max_train_steps}")

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(global_step, train_config.max_train_steps),
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        logger.info(f"Epoch: {epoch} / {num_train_epochs}")

        lora_network.on_epoch_start(text_encoder, unet)

        train_loss = 0.0
        for step, batch in enumerate(dataloader):
            if (step + 1) % 5 == 0:
                break
            with accelerator.accumulate(lora_network):
                # Convert images to latent space.
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents.
                noise = torch.randn_like(latents)
                if train_config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += train_config.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                batch_size = latents.shape[0]
                # Sample a random timestep for each image.
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at
                # each timestep (this is the forward diffusion process).
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type.
                if train_config.prediction_type is not None:
                    # Set the prediction_type of scheduler if it's defined in
                    # train_config.
                    noise_scheduler.register_to_config(
                        prediction_type=train_config.prediction_type
                    )
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        "Unknown prediction type"
                        f" {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual.
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

                # Gather the losses across all processes for logging (if we use
                # distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(train_config.train_batch_size)
                ).mean()
                train_loss += (
                    avg_loss.item() / train_config.gradient_accumulation_steps
                )

                # Backpropagate.
                accelerator.backward(loss)
                if (
                    accelerator.sync_gradients
                    and train_config.max_grad_norm is not None
                ):
                    params_to_clip = lora_network.get_trainable_params()
                    accelerator.clip_grad_norm_(
                        params_to_clip, train_config.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step
            # behind the scenes.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (
                    train_config.save_every_n_steps is not None
                    and global_step % train_config.save_every_n_steps == 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        _save_checkpoint(
                            idx=global_step,
                            prefix="step",
                            out_dir=out_dir,
                            network=accelerator.unwrap_model(lora_network),
                            save_dtype=weight_dtype,
                            train_config=train_config,
                            logger=logger,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= train_config.max_train_steps:
                break

        # Save a checkpoint.
        if (
            train_config.save_every_n_epochs is not None
            and (epoch + 1) % train_config.save_every_n_epochs == 0
        ):
            if accelerator.is_main_process:
                accelerator.wait_for_everyone()
                _save_checkpoint(
                    idx=epoch + 1,
                    prefix="epoch",
                    out_dir=out_dir,
                    network=accelerator.unwrap_model(lora_network),
                    save_dtype=weight_dtype,
                    train_config=train_config,
                    logger=logger,
                )

        # Generate validation images.
        if (
            len(train_config.validation_prompts) > 0
            and (epoch + 1) % train_config.validate_every_n_epochs == 0
        ):
            if accelerator.is_main_process:
                _generate_validation_images(
                    epoch=epoch + 1,
                    out_dir=out_dir,
                    accelerator=accelerator,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    unet=unet,
                    train_config=train_config,
                    logger=logger,
                )

    # End `for epoch in range(first_epoch, num_train_epochs):`

    #     # Save the lora layers
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         unet = unet.to(torch.float32)
    #         unet.save_attn_procs(args.output_dir)

    #         if args.push_to_hub:
    #             save_model_card(
    #                 repo_id,
    #                 images=images,
    #                 base_model=args.pretrained_model_name_or_path,
    #                 dataset_name=args.dataset_name,
    #                 repo_folder=args.output_dir,
    #             )
    #             upload_folder(
    #                 repo_id=repo_id,
    #                 folder_path=args.output_dir,
    #                 commit_message="End of training",
    #                 ignore_patterns=["step_*", "epoch_*"],
    #             )

    # # Final inference
    # # Load previous pipeline
    # pipeline = DiffusionPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     revision=args.revision,
    #     torch_dtype=weight_dtype,
    # )
    # pipeline = pipeline.to(accelerator.device)

    # # load attention processors
    # pipeline.unet.load_attn_procs(args.output_dir)

    # # run inference
    # generator = torch.Generator(device=accelerator.device)
    # if args.seed is not None:
    #     generator = generator.manual_seed(args.seed)
    # images = []
    # for _ in range(args.num_validation_images):
    #     images.append(
    #         pipeline(
    #             args.validation_prompt,
    #             num_inference_steps=30,
    #             generator=generator,
    #         ).images[0]
    #     )

    # if accelerator.is_main_process:
    #     for tracker in accelerator.trackers:
    #         if len(images) != 0:
    #             if tracker.name == "tensorboard":
    #                 np_images = np.stack([np.asarray(img) for img in images])
    #                 tracker.writer.add_images(
    #                     "test", np_images, epoch, dataformats="NHWC"
    #                 )
    #             if tracker.name == "wandb":
    #                 tracker.log(
    #                     {
    #                         "test": [
    #                             wandb.Image(
    #                                 image,
    #                                 caption=f"{i}: {args.validation_prompt}",
    #                             )
    #                             for i, image in enumerate(images)
    #                         ]
    #                     }
    #                 )

    # accelerator.end_training()
