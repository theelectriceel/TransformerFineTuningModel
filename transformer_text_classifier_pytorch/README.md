# Transformer Text Classification with PyTorch and Hugging Face

This project fine-tunes a small Transformer model for text classification using PyTorch and Hugging Face. 

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
