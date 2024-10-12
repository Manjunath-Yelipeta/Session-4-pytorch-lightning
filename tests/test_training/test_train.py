import pytest
import hydra
from pathlib import Path
import rootutils
from omegaconf import DictConfig, OmegaConf
import os

# Setup root directory and print it
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root directory: {root}")

# Import train function
from src.train import train

@pytest.fixture
def config():
    with hydra.initialize(version_base="1.3", config_path='../../configs'):
        cfg = hydra.compose(config_name="train.yaml")

    # Print cfg to YAML format
    print("Configuration in YAML format:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set fast_dev_run to True
    cfg.trainer.fast_dev_run = True
    
    return cfg

def test_dog_breed_training(config: DictConfig, tmp_path: Path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")

    # Run training
    results = train(config)

    # Assertions
    assert results is not None
    assert "config" in results
    assert "model" in results
    assert "trainer" in results

    # Check if log directory was created
    assert Path(config.paths.log_dir).exists()

    # Check if the model was trained
    assert results["trainer"].global_step > 0

    # You can add more specific assertions based on what you expect from a successful run
    # For example:
    # assert "train_loss" in results["trainer"].callback_metrics
    # assert results["trainer"].callback_metrics["train_loss"] < 1.0

if __name__ == "__main__":
    pytest.main([__file__])
