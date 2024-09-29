import lightning as L
import torch
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
from sklearn.metrics import classification_report

log = get_pylogger(__name__)


def eval():
    print("ABC")
    class_names=['Beagle','Bulldog','German_Shepherd','Labrador_Retriever', 'Rottweiler','Boxer','Dachshund','Golden_Retriever','Poodle','Yorkshire_Terrier']
    device = torch.device("cpu")   #"cuda:0"
    #data_module = DogDataModule()
    #model = DogClassifier(lr=1e-3)
    model = DogClassifier().to(device)
    # create model and load state dict
    #model.load_state_dict(torch.load("logs/model_tr.ckpt"))
    model.load_state_dict(torch.load("logs/model_tr.ckpt", map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    y_true=[]
    y_pred=[]
    datamodule = DogDataModule()
    datamodule.setup()
    with torch.no_grad():
        for test_data in datamodule.test_dataloader():
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    print("My report")            
    print(classification_report(y_true,y_pred,target_names=class_names,digits=4))
    config = {"classification report": classification_report(y_true,y_pred,target_names=class_names,digits=4)}
    print_config_tree(config, resolve=True, save_to_file=True)
    
    
    
    
if __name__ == "__main__":
    eval()