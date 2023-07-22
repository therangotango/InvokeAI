import json

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.training.lora.lora_training_config import (
    LoraTrainingConfig,
)


def run_lora_training(
    app_config: InvokeAIAppConfig, train_config: LoraTrainingConfig
):
    logger.info("Starting LoRA Training.")
    logger.info(
        f"Configuration:\n{json.dumps(train_config.dict(), indent=2, default=str)}"
    )

    accelerator_config = ProjectConfiguration()
    accelerator_config.logging_dir = train_config.output_dir / "logs"
    accelerator = Accelerator(
        project_config=accelerator_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=train_config.report_to,
    )
