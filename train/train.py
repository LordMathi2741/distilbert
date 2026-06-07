import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

model_name = "theArijitDas/distilbert-finetuned-fake-reviews"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

df = pd.read_csv("./fake_reviews_es.csv")
df["label"] = df["label"].apply(lambda x: 1 if str(x).strip() == "CG" else 0)

dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.2)

def tokenize_function(examples):
    return tokenizer(examples["text_"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

columnas_mantener = ["input_ids", "attention_mask", "label"]
columnas_eliminar = [col for col in tokenized_dataset["train"].column_names if col not in columnas_mantener]

tokenized_dataset = tokenized_dataset.remove_columns(columnas_eliminar)
tokenized_dataset.set_format(type="torch")

metricas = evaluate.combine(["accuracy", "f1", "recall"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metricas.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    eval_strategy="epoch",       
    save_strategy="epoch",       
    learning_rate=2e-5,          
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,         
    weight_decay=0.01,
    load_best_model_at_end=True, 
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./fake_reviews_es")
tokenizer.save_pretrained("./fake_reviews_es")