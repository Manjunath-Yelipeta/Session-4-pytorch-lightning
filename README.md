# Dog Classifier Pytorch Lighting

## Requirements

Using pip

```bash
pip install -r requirements.txt
```

Using UV

```bash
uv venv
uv pip install -r pyproject.toml
source .venv/bin/activate
```

## Train

```bash
python src/train.py
```

## Tensorboard

```bash
tensorboard --logdir logs
```


## Infer

```bash
python src/infer.py --input_folder samples --output_folder predictions --ckpt_path "/workspace/lightning-template-hydra/logs/dog_classifier/version_5/checkpoints/epoch=0-step=3.ckpt"
```



Dog Breed Classifier
This project is a PyTorch Lightning implementation of a dog breed classifier using a pre-trained ResNet18 model.
Setup

Clone this repository:
Copygit clone https://github.com/yourusername/dog-classifier-pl.git
cd dog-classifier-pl

Install the required packages:
Copypip install -r requirements.txt


Usage
Training
To train the model, run:
Copypython src/train.py
This will download the dataset, train the model, and save the logs and checkpoints in the logs folder.
Evaluation
To evaluate the model on the test set, run:
Copypython src/eval.py
Inference
To run inference on a folder of images:
Copypython src/infer.py --input_folder samples --output_folder predictions --ckpt_path "logs/checkpoints/epoch=0-step=3.ckpt"
Replace the --ckpt_path with the path to your best checkpoint.
Project Structure
Copydog-classifier-pl/
├── README.md
├──Last edited 6 minutes ago

## Black

```bash
black .
```
