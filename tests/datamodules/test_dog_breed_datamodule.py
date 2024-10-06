# tests/test_dog_breed_data_module.py
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pytest
from src.datamodules.dog_breed import DogBreedDataModule
import os
import json
from pathlib import Path
import torch

@pytest.fixture(scope="module")
def datamodule():
    return DogBreedDataModule(
        gdrive_url="https://drive.google.com/file/d/15Yk-AaufO41Ocs-mnbgBCW46SsQRFlAI/view?usp=sharing",
        dl_path="./data",
        batch_size=4,
        num_workers=0,
        train_val_test_split=(0.8, 0.1, 0.1),
    )

def test_datamodule_attributes(datamodule):
    assert isinstance(datamodule.gdrive_url, str)
    assert isinstance(datamodule._dl_path, Path)
    assert isinstance(datamodule.batch_size, int)
    assert isinstance(datamodule.num_workers, int)
    assert isinstance(datamodule.train_val_test_split, tuple)
    assert len(datamodule.train_val_test_split) == 3
    assert sum(datamodule.train_val_test_split) == 1.0

def test_prepare_data(datamodule):
    datamodule.prepare_data()
    assert Path(datamodule._dl_path).exists(), "Data directory not created"
    assert Path(datamodule.split_file).exists(), "Split file not created"

def test_setup(datamodule):
    datamodule.setup()
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert datamodule.num_classes > 0
    assert len(datamodule.class_names) > 0

def test_train_dataloader(datamodule):
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert len(train_dataloader.dataset) > 0

def test_val_dataloader(datamodule):
    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)
    assert len(val_dataloader.dataset) > 0

def test_test_dataloader(datamodule):
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()
    assert isinstance(test_dataloader, torch.utils.data.DataLoader)
    assert len(test_dataloader.dataset) > 0

def test_dataloader_output_shape(datamodule):
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert x.shape == (datamodule.batch_size, 3, 224, 224)
    assert y.shape == (datamodule.batch_size,)

@pytest.mark.parametrize("stage", [None, "fit", "test"])
def test_setup_stage(datamodule, stage):
    datamodule.setup(stage=stage)
    if stage in (None, "fit"):
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
    if stage in (None, "test"):
        assert datamodule.test_dataset is not None

def test_transforms(datamodule):
    assert datamodule.train_transform is not None
    assert datamodule.valid_transform is not None
    assert len(datamodule.train_transform.transforms) == 7
    assert len(datamodule.valid_transform.transforms) == 4

def test_class_names_property(datamodule):
    datamodule.setup()
    assert isinstance(datamodule.class_names, list)
    assert len(datamodule.class_names) > 0
    assert all(isinstance(name, str) for name in datamodule.class_names)

def test_num_classes_property(datamodule):
    datamodule.setup()
    assert isinstance(datamodule.num_classes, int)
    assert datamodule.num_classes > 0
    assert datamodule.num_classes == len(datamodule.class_names)

def test_data_split_sizes(datamodule):
    datamodule.setup()
    total_size = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
    assert len(datamodule.train_dataset) / total_size == pytest.approx(datamodule.train_val_test_split[0], rel=1e-2)
    assert len(datamodule.val_dataset) / total_size == pytest.approx(datamodule.train_val_test_split[1], rel=1e-2)
    assert len(datamodule.test_dataset) / total_size == pytest.approx(datamodule.train_val_test_split[2], rel=1e-2)