import os
from pathlib import Path

import rootutils

# Setup the root of the project
root = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf

from src.datamodules.dog_breed import DogBreedDataModule
from src.models.catdog_classifier import DogClassifier
from src.utils.utils import task_wrapper
from src.utils.pylogger import get_pylogger
from src.utils.rich_utils import print_config_tree, print_rich_panel

log = get_pylogger(__name__)

@hydra.main(config_path=str(root / "configs"), config_name="eval.yaml", version_base="1.3")
@task_wrapper
def evaluate(cfg: DictConfig) -> None:
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    # Determine the checkpoint path
    ckpt_path = hydra.utils.instantiate(cfg.ckpt_path)
    log.info(f"Checkpoint path: {ckpt_path}")

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Print path to best checkpoint
    log.info(f"Best model checkpoint: {ckpt_path}")

    # Print the configuration
    print_rich_panel("Configuration", "Evaluation Config")
    print(OmegaConf.to_yaml(cfg))

    return trainer.callback_metrics

if __name__ == "__main__":
    evaluate()
