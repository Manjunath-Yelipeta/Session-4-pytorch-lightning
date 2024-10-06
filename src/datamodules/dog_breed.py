import os
import gdown
import zipfile
import json
from pathlib import Path
import lightning as L
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from typing import Union, Optional, List
import torch

class DogBreedDataModule(L.LightningDataModule):
    def __init__(self, 
                 gdrive_url: str,
                 dl_path: Union[str, Path] = "./data_dir", 
                 num_workers: int = 0, 
                 batch_size: int = 8,
                 train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
                 seed: int = 42,
                 pin_memory: bool = False):  # Add pin_memory parameter
        super().__init__()
        self._dl_path = Path(dl_path)
        self._num_workers = num_workers
        self._batch_size = batch_size
        self.gdrive_url = gdrive_url
        self.train_val_test_split = train_val_test_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split_file = self._dl_path / 'split_indices.json'
        self.seed = seed
        self.pin_memory = pin_memory  # Store pin_memory
        self._class_names: Optional[List[str]] = None
        self._num_classes: Optional[int] = None

    def prepare_data(self):
        """Download images, prepare datasets, and create splits."""
        try:
            # Create data_dir if it doesn't exist
            self._dl_path.mkdir(parents=True, exist_ok=True)

            # Download and extract dataset if not already done
            if not (self._dl_path / "dataset").exists():
                self._download_and_extract_dataset()

            # Create and save splits if not already done
            if not self.split_file.exists():
                self._create_and_save_splits()

        except Exception as e:
            print(f"An error occurred: {e}")

    def _download_and_extract_dataset(self):
        # Parse the file ID from the Google Drive URL
        file_id = gdown.parse_url.parse_url(self.gdrive_url)[0]

        # Set the output path for the downloaded zip file
        output_path = self._dl_path / 'downloaded_file.zip'

        # Download the file
        gdown.download(id=file_id, output=str(output_path), quiet=False)

        # Extract the contents of the zip file
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(self ._dl_path)

        # Remove the zip file after extraction
        os.remove(output_path)

        print(f"Dataset successfully downloaded and extracted to {self._dl_path}")

    def _create_and_save_splits(self):
        full_dataset = ImageFolder(root=self.data_path)
        total_samples = len(full_dataset)
        
        train_samples = int(self.train_val_test_split[0] * total_samples)
        val_samples = int(self.train_val_test_split[1] * total_samples)
        test_samples = total_samples - train_samples - val_samples

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_samples, val_samples, test_samples],
            generator=torch.Generator().manual_seed(self.seed)
        )

        splits = {
            'train': train_dataset.indices,
            'val': val_dataset.indices,
            'test': test_dataset.indices
        }
        with open(self.split_file, 'w') as f:
            json.dump(splits, f)

        self.logger.info(f"Split indices saved to {self.split_file}")

    @property
    def data_path(self):
        return self._dl_path / "dataset"

    def setup(self, stage: Optional[str] = None):
        """Create train, validation, and test datasets using saved splits."""
        if self.test_dataset is None or stage == "test":
            full_dataset = ImageFolder(root=self.data_path, transform=self.valid_transform)
            
            with open(self.split_file, 'r') as f:
                splits = json.load(f)

            self.test_dataset = Subset(full_dataset, splits['test'])

        if stage != "test":  # For "fit" or None
            if self.train_dataset is None or self.val_dataset is None:
                full_dataset = ImageFolder(root=self.data_path, transform=self.train_transform)
                
                with open(self.split_file, 'r') as f:
                    splits = json.load(f)

                self.train_dataset = Subset(full_dataset, splits['train'])
                self.val_dataset = Subset(full_dataset, splits['val'])
                self.val_dataset.dataset.transform = self.valid_transform

        # Set class names and num_classes
        if self._class_names is None:
            self._class_names = full_dataset.classes
            self._num_classes = len(self._class_names)

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    @property
    def class_names(self) -> List[str]:
        if self._class_names is None:
            raise ValueError("setup() must be called before accessing class_names")
        return self._class_names

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            raise ValueError("setup() must be called before accessing num_classes")
        return self._num_classes