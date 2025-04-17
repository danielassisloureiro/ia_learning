#Transformer (Simplificado) – Classificação de Texto com BERT
#intstalar -> pip install transformers datasets
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import load_dataset
import tensorflow as tf

# Dataset IMDb
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

encoded = dataset.map(tokenize, batched=True)
encoded.set_format("tensorflow", columns=["input_ids", "attention_mask", "label"])

train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": encoded["train"]["input_ids"], "attention_mask": encoded["train"]["attention_mask"]},
    encoded["train"]["label"]
)).batch(16)

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=1)