import pickle
from config import ENCODER_PATH
# Import your original XGBoost predict functions
from predict import load_trained_model as load_xgb_model, predict_text as predict_xgb
# Import your new BERT predict functions
from predict_bert import load_bert_model, predict_text as predict_bert

def main():
    print("Loading models into memory (this might take a few seconds)...\n")
    
    # 1. Load XGBoost Setup
    xgb_model = load_xgb_model()
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    # 2. Load BERT Setup
    bert_model, tokenizer, bert_classes = load_bert_model()

    # 3. Define your real-world test cases
    # Try to use phrasing that IS NOT exactly how it appears in your CSV
    test_tickets = [
        "I've been locked out of my account since yesterday and the reset link isn't arriving.",
        "Is there a way to download all my billing history into an excel file?",
        "Every time I click the submit button on the dashboard, the page crashes and gives me a 500 error.",
        "Can you add a dark mode feature? The current screen is too bright."
    ]

    print("=============================================")
    print("          MODEL COMPARISON TEST              ")
    print("=============================================")

    for i, ticket in enumerate(test_tickets, 1):
        print(f"\nTest Case {i}:")
        print(f"User Input: \"{ticket}\"")
        
        # Get XGBoost Prediction
        xgb_prediction = predict_xgb(xgb_model, ticket, encoder)
        print(f"XGBoost Prediction: {xgb_prediction}")

        # Get BERT Prediction
        bert_prediction = predict_bert(ticket, bert_model, tokenizer, bert_classes)
        print(f"BERT Prediction:    {bert_prediction}")
        print("-" * 45)

if __name__ == "__main__":
    main()