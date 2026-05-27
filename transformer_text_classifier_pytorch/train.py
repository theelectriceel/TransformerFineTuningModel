'''
Fine-tune a Transformer model for text classification.

Dataset: AG News
Model: distilbert-base-uncased
'''

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "ag_news"
OUTPUT_DIR = "./saved_model"


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    train_dataset = dataset["train"].shuffle(seed=21).select(range(8000))
    test_dataset = dataset["test"].shuffle(seed=21).select(range(2000))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
    )

    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

    print("Training model...")
    trainer.train()

    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
