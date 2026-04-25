import numpy as np
import torch
import torch.nn.functional as F
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_OUTPUT_DIR = "/content/drive/MyDrive/Ticket_classification_Project/distilbert_ticket_classifier"

def load_distilbert_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR)
    classes = np.load(os.path.join(MODEL_OUTPUT_DIR, 'label_classes.npy'), allow_pickle=True)
    return model, tokenizer, classes

def predict_text(text, model, tokenizer, classes):
    model.eval() 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # --- UPDATE 3: Calculate Top 3 Probabilities ---
    logits = outputs.logits
    # Convert raw logits to percentages (probabilities)
    probabilities = F.softmax(logits, dim=1)
    
    # Get the top 3 highest probabilities and their corresponding indices
    top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
    
    # Format the results into a clean list of dictionaries
    results = []
    for i in range(3):
        confidence = top_probs[0][i].item() * 100
        predicted_class = classes[top_indices[0][i].item()]
        results.append({
            "issue": predicted_class,
            "confidence": round(confidence, 1)
        })
        
    return results
    # -----------------------------------------------

# Example of how to print the new results if running this file directly:
if __name__ == "__main__":
    distil_model, distil_tokenizer, distil_classes = load_distilbert_model()
    
    test_ticket = "I was messing around with the URL parameters in the dashboard and noticed I could see another organization's private data if I just change the ID number."
    
    predictions = predict_text(test_ticket, distil_model, distil_tokenizer, distil_classes)
    
    print(f"\nTicket: '{test_ticket}'")
    print("\nTop 3 Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['issue']} ({pred['confidence']}%)")