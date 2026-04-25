from predict import load_trained_model, load_encoder, predict_priority

print("Loading model and encoder...")
model = load_trained_model()
encoder = load_encoder()

sentence = "I noticed a suspicious login on my account."

predicted_priority = predict_priority(model, sentence, encoder)

print(f"\nInput Message: {sentence}")
print(f"Predicted Priority: {predicted_priority}")