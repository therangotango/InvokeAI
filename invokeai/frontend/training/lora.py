import os
from pathlib import Path

import yaml

from invokeai.app.services.config import InvokeAIAppConfig, PagingArgumentParser
from invokeai.backend.training.lora.lora_training import run_lora_training
from invokeai.backend.training.lora.lora_training_config import (
    LoraTrainingConfig,
)


def parse_args():
    config = InvokeAIAppConfig.get_config()

    parser = PagingArgumentParser(description="LoRA model training.")

    parser.add_argument(
        "--cfg_file",
        type=Path,
        required=True,
        help=(
            "Path to the YAML training config file. See `LoraTrainingConfig`"
            " for the supported fields."
        ),
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        # TODO(ryand): Decide on a training directory structure and update for
        # consistency with TI training.
        default=os.path.join(config.root, "training/lora/output"),
        help=(
            "The output directory where the training outputs (model"
            " checkpoints, logs, intermediate predictions) will be written."
            " Defaults to `$INVOKEAI_HOME/training/lora/output`."
        ),
    )

    return parser.parse_args()


def main():
    app_config = InvokeAIAppConfig.get_config()
    args = parse_args()

    # Load YAML config file.
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Override 'output_dir' config.
    cfg["base_output_dir"] = args.base_output_dir

    train_config = LoraTrainingConfig(**cfg)
    run_lora_training(app_config, train_config)


if __name__ == "__main__":
    main()
