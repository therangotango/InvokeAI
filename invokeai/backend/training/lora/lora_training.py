import json
import logging
import os
import random

import datasets
import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torchvision import transforms
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


def _initialize_accelerator(train_config: LoraTrainingConfig) -> Accelerator:
    """Configure Hugging Face accelerate and return an Accelerator.

    Args:
        train_config (LoraTrainingConfig): LoRA training configuration.

    Returns:
        Accelerator
    """
    accelerator_project_config = ProjectConfiguration(
        project_dir=train_config.output_dir,
        logging_dir=train_config.output_dir / "logs",
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

    # Move models to device, and cast dtype.
    weight_dtype = torch.float32
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


def run_lora_training(
    app_config: InvokeAIAppConfig, train_config: LoraTrainingConfig
):
    accelerator = _initialize_accelerator(train_config)
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

    tokenizer, noise_scheduler, text_encoder, vae, unet = _load_models(
        accelerator, app_config, train_config, logger
    )

    if train_config.xformers:
        import xformers

        unet.enable_xformers_memory_efficient_attention()
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
    lr_scheduler = get_scheduler(
        train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps
        * train_config.gradient_accumulation_steps,
        num_training_steps=train_config.max_train_steps
        * train_config.gradient_accumulation_steps,
    )

    (
        unet,
        text_encoder,
        lora_network,
        optimizer,
        dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        text_encoder,
        lora_network,
        optimizer,
        data_loader,
        lr_scheduler,
    )

    x = train_features, train_labels = next(iter(data_loader))
    logger.info(x.keys())
    logger.info(x["pixel_values"].shape)
    logger.info(x["input_ids"].shape)
