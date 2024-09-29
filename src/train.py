import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
)

from datamodules.catdog import DogDataModule
from models.catdog_classifier import DogClassifier
from utils.utils import task_wrapper
from utils.pylogger import get_pylogger
from utils.rich_utils import print_config_tree, print_rich_progress, print_rich_panel

log = get_pylogger(__name__)


@task_wrapper
def train():
    # Set up data module
    data_module = DogDataModule()

    # Set up model
    model = DogClassifier(lr=1e-3)

    # Set up logger
    logger = TensorBoardLogger(save_dir="logs", name="dog_classification")

    # Set up callbacks
    #checkpoint_callback = ModelCheckpoint(monitor="val/loss",save_on_train_epoch_end=True,dirpath='/workspace/lightning-template-hydra/logs',filename="model.pt")
    checkpoint_callback = ModelCheckpoint(save_on_train_epoch_end=True,dirpath='/workspace/lightning-template-hydra/logs',filename="model_tr")
    rich_progress_bar = RichProgressBar()
    rich_model_summary = RichModelSummary(max_depth=2)

    # Set up trainer
    trainer = L.Trainer(
        limit_train_batches=0.05,
        limit_val_batches=0.05,
        max_epochs=1,
        callbacks=[checkpoint_callback, rich_progress_bar, rich_model_summary],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # Print config
    config = {"data": vars(data_module), "model": vars(model), "trainer": vars(trainer)}
    print_config_tree(config, resolve=True, save_to_file=True)

    # Train the model
    print_rich_panel("Starting training", "Training")
    trainer.fit(model, datamodule=data_module)

    # Test the model
    print_rich_panel("Starting testing", "Testing")
    trainer.test(model, datamodule=data_module)

    print_rich_progress("Finishing up")


if __name__ == "__main__":
    train()
