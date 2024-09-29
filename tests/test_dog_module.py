import pytest
from src.datamodules import catdog
from src.datamodules.catdog import DogDataModule


def test_catdog_datamodule():
    datamodule = DogDataModule()

    # Test prepare_data
    #datamodule.prepare_data()

    # Test setup
    datamodule.setup()

    # Test dataloaders
    train_loader = datamodule.train_dataloader()
    #val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Add assertions to check if the dataloaders are correctly set up
    assert len(train_loader) > 0
    #assert len(val_loader) > 0
    assert len(test_loader) > 0
