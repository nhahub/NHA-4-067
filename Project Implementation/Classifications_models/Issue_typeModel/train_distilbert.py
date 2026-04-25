import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# --- Hardcoded Paths for Google Colab / Drive ---
DATA_PATH = "/content/drive/MyDrive/Ticket_classification_Project/Final_version.csv"
MODEL_OUTPUT_DIR = "/content/drive/MyDrive/Ticket_classification_Project/distilbert_ticket_classifier"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# --- Custom Trainer for Class Weights ---
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert the numpy array of weights to a PyTorch tensor and move it to the correct device (GPU/CPU)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract the labels from the inputs
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply the heavy penalties for missing minority classes
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train_distilbert():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH).dropna(subset=['initial_message', 'refined_issue'])
    
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['refined_issue'])
    num_classes = len(encoder.classes_)

    print("Calculating class weights to balance the dataset mathematically...")
    # This automatically assigns higher weights to rare categories (like security_vulnerability) 
    # and lower weights to common ones (like performance_timeout)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    
    # Split the original, un-duplicated data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['initial_message'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42
    )

    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    print("Tokenizing texts...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=num_classes
    )

    # 5 Epochs is recommended since the model needs time to adjust to the new penalty weights
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=5,              
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    print("Initializing Custom Weighted Trainer...")
    trainer = WeightedTrainer(
        class_weights=class_weights,  # Passing our calculated weights here
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Starting fine-tuning...")
    trainer.train()
    
    print("Saving model and tokenizer...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    # Save the classes so the prediction script knows how to translate IDs back to text
    np.save(os.path.join(MODEL_OUTPUT_DIR, 'label_classes.npy'), encoder.classes_)
    print("Training complete and saved to Google Drive!")

if __name__ == "__main__":
    train_distilbert()