import json
import logging

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger, MultiProcessAdapter
from accelerate.utils import ProjectConfiguration, set_seed

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.training.lora.lora_training_config import LoraTrainingConfig


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


def run_lora_training(app_config: InvokeAIAppConfig, train_config: LoraTrainingConfig):
    accelerator = _initialize_accelerator(train_config)
    logger = _initialize_logging(accelerator)

    # Log the accelerator configuration from every process to help with
    # debugging.
    logger.info(accelerator.state, main_process_only=False)

    logger.info("Starting LoRA Training.")
    logger.info(
        f"Configuration:\n{json.dumps(train_config.dict(), indent=2, default=str)}"
    )
