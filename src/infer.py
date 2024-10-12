import os
from pathlib import Path
import rootutils
import numpy as np

# Setup the root of the project
root = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.models.catdog_classifier import DogClassifier
from src.utils.rich_utils import print_config_tree
from src.utils import utils

log = utils.get_pylogger(__name__)

def load_model(cfg: DictConfig):
    ckpt_path = hydra.utils.instantiate(cfg.ckpt_path)
    log.info(f"Loading model from {ckpt_path}")
    model = DogClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def process_image(image_path, transform):
    img = Image.open(image_path)
    img_for_model = transform(img.convert("RGB")).unsqueeze(0)
    return img, img_for_model

def get_prediction(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

def save_prediction(img, predicted_label, confidence, output_path):
    # Convert PIL Image to numpy array in RGB
    img_array = np.array(img.convert("RGB"))
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    # Display the image
    ax.imshow(img_array)
    ax.axis('off')
    
    # Add title with prediction and confidence
    ax.set_title(f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}", fontsize=16, pad=20)
    
    # Add a subtle border around the image
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
        spine.set_linewidth(2)
    
    # Adjust layout and save
    plt.tight_layout(pad=3)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.5, format='png')
    
    # Close the figure to free up memory
    plt.close(fig)

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

@hydra.main(config_path=str(root / "configs"), config_name="infer.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Print configuration
    print_config_tree(cfg, resolve=True, save_to_file=True)

    # Load model
    model = load_model(cfg)

    # Instantiate the data module
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Setup only the test stage
    datamodule.setup(stage="test")

    # Get transform and class names from the data module
    transform = datamodule.valid_transform
    class_names = datamodule.class_names

    # Use absolute paths for input and output directories
    input_folder = Path(cfg.infer_paths.input_dir).resolve()
    output_folder = Path(cfg.infer_paths.output_dir).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    log.info(f"Input folder: {input_folder}")
    log.info(f"Output folder: {output_folder}")

    input_samples = list(input_folder.glob("*"))
    log.info(f"Number of input samples for inference: {len(input_samples)}")

    for img_path in input_samples:
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            try:
                img, img_tensor = process_image(img_path, transform)
                predicted_class, confidence = get_prediction(model, img_tensor)
                predicted_label = class_names[predicted_class] if class_names else f"Class_{predicted_class}"

                output_image_path = output_folder / f"{img_path.stem}_pred_{predicted_label}_conf_{confidence:.2f}.png"
                save_prediction(img, predicted_label, confidence, output_image_path)

                log.info(f"Processed {img_path.name}: Predicted {predicted_label} with confidence {confidence:.2f}")
                log.info(f"Saved prediction image to {output_image_path}")
            except Exception as e:
                log.error(f"Error processing {img_path.name}: {str(e)}")

    log.info(f"Inference completed. Results saved in {output_folder}")

if __name__ == "__main__":
    main()
