# Install library yang dibutuhkan
!pip install transformers datasets

# Import library yang diperlukan
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Muat tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModelForSequenceClassification.from_pretrained("medicalai/ClinicalBERT")

# Load dataset
ds = load_dataset("celikmus/symptom_text_to_disease_01")
train_dataset = ds['train']
test_dataset = ds['test']

# Daftar gejala sebagai contoh
symptom_mapping = {
    0: "emotional pain", 1: "hair falling out", 2: "heart hurts", 3: "infected wound",
    4: "foot ache", 5: "shoulder pain", 6: "injury from sports", 7: "skin issue",
    8: "stomach ache", 9: "knee pain", 10: "joint pain", 11: "hard to breath",
    12: "head ache", 13: "body feels weak", 14: "feeling dizzy", 15: "back pain",
    16: "open wound", 17: "internal pain", 18: "blurry vision", 19: "acne",
    20: "muscle pain", 21: "neck pain", 22: "cough", 23: "ear ache", 24: "feeling cold",
}

# Tambahkan kolom baru dengan nama gejala
for entry in train_dataset:
    entry['symptom_name'] = symptom_mapping[entry['labels']]

# Tampilkan contoh data
print(f"Teks: {train_dataset[0]['text']}, Nama Gejala: {train_dataset[0]['symptom_name']}")

# Fungsi untuk memproses data
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

# Terapkan fungsi pemrosesan pada dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Pastikan dataset memiliki kolom input_ids, attention_mask, dan labels
print(train_dataset.column_names)

# Ambil semua label dari dataset dan hitung jumlah label unik
labels = train_dataset['labels']
unique_labels = set(labels)
num_labels = len(unique_labels)

# Muat model dengan jumlah label yang benar
model = AutoModelForSequenceClassification.from_pretrained("medicalai/ClinicalBERT", num_labels=num_labels)

# Tentukan argumen pelatihan
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',  # Evaluasi setiap epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Fungsi untuk menghitung metrik
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# Buat trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Tambahkan fungsi compute_metrics di sini
)

# Mulai pelatihan
trainer.train()

# Lakukan evaluasi
results = trainer.evaluate()
print(results)
