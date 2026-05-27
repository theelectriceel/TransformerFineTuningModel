
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "./saved_model"

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def predict(text: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    return LABELS[predicted_class]


if __name__ == "__main__":
    sample_text = (
        "Researchers released a new artificial intelligence model "
        "that improves performance on language understanding tasks."
    )

    prediction = predict(sample_text)
    print(f"Text: {sample_text}")
    print(f"Predicted class: {prediction}")
