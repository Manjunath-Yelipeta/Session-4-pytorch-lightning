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
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report

from src.datamodules.dog_breed import DogBreedDataModule
from models.catdog_classifier import DogClassifier
from utils.utils import task_wrapper
from utils.pylogger import get_pylogger
from utils.rich_utils import print_config_tree, print_rich_progress, print_rich_panel

log = get_pylogger(__name__)

@hydra.main(config_path=str(root / "configs"), config_name="eval.yaml", version_base="1.3")
@task_wrapper
def eval(cfg: DictConfig):
    # Resolve the configuration
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    # Print the configuration
    print_rich_panel("Configuration", "Evaluation Config")
    print(OmegaConf.to_yaml(cfg))

    # Debug logging
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Configuration keys: {cfg.keys()}")

    # Set up data module
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup(stage="test")  # Only setup for test stage

    # Set up model
    model = DogClassifier.load_from_checkpoint(cfg.model.checkpoint_path)
    model.eval()

    device = torch.device(cfg.hardware.device)
    model = model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=data_module.class_names, digits=4)
    
    print_rich_panel("Classification Report", "Evaluation Results")
    print(report)

    # Save the report
    config = {"classification_report": report}
    print_config_tree(config, resolve=True, save_to_file=True)

    return {"config": cfg, "classification_report": report}

if __name__ == "__main__":
    eval()