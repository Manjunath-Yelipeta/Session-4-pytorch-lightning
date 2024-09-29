import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.catdog_classifier import DogClassifier
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.console import Console

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

console = Console()


def inference(model, image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply the transform to the image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the same device as the model
    img_tensor = img_tensor.to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Map the predicted class to the label
    #class_labels = ["cat", "dog"]  # Assuming 0 is cat and 1 is dog
    class_labels =['Beagle','Bulldog','German_Shepherd','Labrador_Retriever', 'Rottweiler','Boxer','Dachshund','Golden_Retriever','Poodle','Yorkshire_Terrier']
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    return img, predicted_label, confidence


def save_prediction(img, predicted_label, confidence, output_path):
    # Create the figure and display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Predicted: {predicted_label.capitalize()} (Confidence: {confidence:.2f})"
    )

    # Save the figure
    plt.savefig(output_path)
    plt.close()


def main(args):
    console.print(Panel("Starting inference", title="Inference", expand=False))

    # Load the model
    #model = CatDogClassifier.load_from_checkpoint(args.ckpt_path)

    #model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cpu")   #"cuda:0"
    #data_module = DogDataModule()
    #model = DogClassifier(lr=1e-3)
    model = DogClassifier().to(device)
    # create model and load state dict
    #model.load_state_dict(torch.load("logs/model_tr.ckpt"))
    model.load_state_dict(torch.load("logs/model_tr.ckpt", map_location=torch.device('cpu'))['state_dict'])
    # Create the predictions folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(args.input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    with Progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for filename in image_files:
            image_path = os.path.join(args.input_folder, filename)
            img, predicted_label, confidence = inference(model, image_path)

            # Save the prediction image
            output_image_path = os.path.join(
                args.output_folder, f"{os.path.splitext(filename)[0]}_prediction.png"
            )
            save_prediction(img, predicted_label, confidence, output_image_path)

            # Save the prediction text
            output_text_path = os.path.join(
                args.output_folder, f"{os.path.splitext(filename)[0]}_prediction.txt"
            )
            with open(output_text_path, "w") as f:
                f.write(f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}")

            progress.update(task, advance=1, description=f"[green]Processed {filename}")

    console.print(Panel("Inference completed", title="Finished", expand=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on images")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="predictions",
        help="Path to the folder to save predictions",
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the model checkpoint"
    )

    args = parser.parse_args()
    main(args)
