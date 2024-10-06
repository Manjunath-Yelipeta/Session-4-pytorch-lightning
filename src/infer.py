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
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision import transforms
import random
import shutil

# Setup the root of the project
root = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

from src.models.catdog_classifier import DogClassifier
from src.datamodules.dog_breed import DogBreedDataModule
from src.utils.rich_utils import print_config_tree
from src.utils import utils


log = utils.get_pylogger(__name__)

def load_model(cfg: DictConfig):
    ckpt_path = cfg.ckpt_path
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

def create_input_folder(cfg: DictConfig, data_module: DogBreedDataModule):
    input_folder = Path(cfg.infer_paths.input_dir)
    total_samples = cfg.total_samples  # This should be 10 in your config
    log.info(f"Attempting to create {total_samples} random samples across all classes")

    if not input_folder.exists():
        input_folder.mkdir(parents=True, exist_ok=True)
        test_dataset = data_module.test_dataset
        class_names = data_module.class_names
        
        log.info(f"Total number of classes: {len(class_names)}")
        
        all_samples = test_dataset.dataset.samples
        random.shuffle(all_samples)
        
        valid_samples = 0
        for original_path, class_idx in all_samples:
            if valid_samples >= total_samples:
                break
            
            class_name = class_names[class_idx]
            
            if is_valid_image(original_path):
                img_path = input_folder / f"{class_name}_{valid_samples}{Path(original_path).suffix}"
                shutil.copy(original_path, img_path)
                valid_samples += 1
                log.info(f"Copied valid image: {img_path}")
        
        log.info(f"Created input folder with {valid_samples} valid samples at {input_folder}")
    else:
        log.info(f"Input folder already exists at {input_folder}")
        existing_samples = list(input_folder.glob("*"))
        log.info(f"Found {len(existing_samples)} existing samples")
    
    # Verify existing images in the input folder
    for img_path in input_folder.glob("*"):
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            if not is_valid_image(img_path):
                log.warning(f"Removing invalid image from input folder: {img_path}")
                img_path.unlink()

    final_sample_count = len(list(input_folder.glob("*")))
    log.info(f"Final number of samples in input folder: {final_sample_count}")

@hydra.main(config_path=str(root / "configs"), config_name="infer.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Print configuration
    print_config_tree(cfg, resolve=True, save_to_file=True)
    log.info(f"Total samples: {cfg.total_samples}")

    # Load model
    model = load_model(cfg)

    # Setup data module
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup(stage="test")

    # Create input folder with samples from test set if it doesn't exist
    create_input_folder(cfg, data_module)

    # Create output folder
    output_folder = Path(cfg.infer_paths.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Perform inference
    transform = data_module.valid_transform
    class_names = data_module.class_names

    input_folder = Path(cfg.infer_paths.input_dir)
    input_samples = list(input_folder.glob("*"))
    log.info(f"Number of input samples for inference: {len(input_samples)}")

    for img_path in input_samples:
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            try:
                img, img_tensor = process_image(img_path, transform)
                predicted_class, confidence = get_prediction(model, img_tensor)
                predicted_label = class_names[predicted_class]

                # Extract the original class name from the filename
                original_class = img_path.stem.split('_')[0]

                output_image_path = output_folder / f"{original_class}_pred_{predicted_label}_conf_{confidence:.2f}.png"
                save_prediction(img, predicted_label, confidence, output_image_path)

                output_text_path = output_folder / f"{original_class}_pred_{predicted_label}_conf_{confidence:.2f}.txt"
                with open(output_text_path, "w") as f:
                    f.write(f"Original: {original_class}\nPredicted: {predicted_label}\nConfidence: {confidence:.2f}")

                log.info(f"Processed {img_path.name}: Original {original_class}, Predicted {predicted_label} with confidence {confidence:.2f}")
            except Exception as e:
                log.error(f"Error processing {img_path.name}: {str(e)}")

    log.info(f"Inference completed. Results saved in {output_folder}")

if __name__ == "__main__":
    main()
