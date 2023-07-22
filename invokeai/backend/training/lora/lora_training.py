import json
import logging

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.training.lora.lora_training_config import LoraTrainingConfig


def run_lora_training(app_config: InvokeAIAppConfig, train_config: LoraTrainingConfig):
    # Configure accelerator.
    accelerator_project_config = ProjectConfiguration(
        project_dir=train_config.output_dir,
        logging_dir=train_config.output_dir / "logs",
    )
    accelerator = Accelerator(
        project_config=accelerator_project_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=train_config.report_to,
    )

    # Configure logging.
    logger = get_logger(__name__)
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

    # Log the accelerator configuration from every process to help with
    # debugging.
    logger.info(accelerator.state, main_process_only=False)

    logger.info("Starting LoRA Training.")
    logger.info(
        f"Configuration:\n{json.dumps(train_config.dict(), indent=2, default=str)}"
    )
