# Transformer Text Classification with PyTorch and Hugging Face

This project fine-tunes a small Transformer model for text classification using PyTorch and Hugging Face. It demonstrates a complete NLP training workflow: dataset loading, tokenization, model fine-tuning, evaluation, and prediction.

## Why this project matters

The goal is to build a clean, reproducible text classification pipeline rather than just call a hosted API. The project covers practical machine learning engineering skills such as:

- PyTorch-backed model training
- Hugging Face Transformers
- Tokenization and dataset preprocessing
- Train/validation splits
- Accuracy and F1 evaluation
- Saving and reloading trained models
- Running inference on new text

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- pandas

## Project structure

```text
transformer_text_classifier_pytorch/
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

## Train the model

This example uses the `ag_news` dataset, a standard text classification dataset with four news categories.

```bash
python train.py
```

The trained model will be saved to:

```text
./saved_model
```

## Run prediction

```bash
python predict.py
```

Example prediction text is included in `predict.py`.

## Skills demonstrated

This project demonstrates beginner-to-intermediate experience with modern NLP model training workflows. It uses a Transformer model with a PyTorch backend and includes preprocessing, training, evaluation, and inference steps.
