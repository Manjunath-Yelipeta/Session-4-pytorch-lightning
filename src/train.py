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

# Now you can import your modules
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
)
import hydra
from omegaconf import DictConfig,OmegaConf

from src.datamodules.dog_breed import DogBreedDataModule
from models.catdog_classifier import DogClassifier
from utils.utils import task_wrapper
from utils.pylogger import get_pylogger
from utils.rich_utils import print_config_tree, print_rich_progress, print_rich_panel

# Setup logging
log = get_pylogger(__name__)

@hydra.main(config_path=str(root / "configs"), config_name="train.yaml", version_base="1.3")
@task_wrapper
def train(cfg: DictConfig):
    # Resolve the configuration
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # Debug logging
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Configuration keys: {cfg.keys()}")
    log.info(f"Paths configuration: {OmegaConf.to_yaml(cfg.paths)}")
    log.info(f"Data configuration: {OmegaConf.to_yaml(cfg.data)}")
    
    # Ensure log directory exists
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Log directory: {log_dir}")

    # Set up data module
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    # Log dataset sizes
    log.info(f"Train Dataset Size: {len(data_module.train_dataset)}")
    log.info(f"Validation Dataset Size: {len(data_module.val_dataset)}")
    log.info(f"Test Dataset Size: {len(data_module.test_dataset)}")
    log.info(f"Class Names: {data_module.class_names}")

    # Set up model
    model = DogClassifier(lr=1e-3)

    # Set up logger
    logger = TensorBoardLogger(save_dir=str(log_dir), name="dog_classification")

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        save_on_train_epoch_end=True,
        dirpath=str(log_dir),
        filename="model_tr",
    )
    rich_progress_bar = RichProgressBar()
    rich_model_summary = RichModelSummary(max_depth=2)

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=2,  # Changed to 2 epochs
        callbacks=[checkpoint_callback, rich_progress_bar, rich_model_summary],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # Print config
    config = {"data": vars(data_module), "model": vars(model), "trainer": vars(trainer)}
    print_config_tree(config, resolve=True, save_to_file=True, log_dir=str(log_dir))

    # Train the model
    print_rich_panel("Starting training", "Training")
    trainer.fit(model, datamodule=data_module)

    # Test the model
    print_rich_panel("Starting testing", "Testing")
    trainer.test(model, datamodule=data_module)

    print_rich_progress("Finishing up")

    return {"config": config, "model": model, "trainer": trainer}

if __name__ == "__main__":
    train()
