import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanMetric, MaxMetric
import timm


class DogClassifier(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

        # Load pre-trained ResNet18 model
        self.model = timm.create_model("resnet18", pretrained=True, num_classes=10)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters(logger=False)

        # Multi-class accuracy with num_classes=2
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
