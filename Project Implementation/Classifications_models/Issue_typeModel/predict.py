import numpy as np
import pickle
from config import MODEL_PATH

def load_trained_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model



def predict_text(model, text, encoder):

    text_list = [text] if isinstance(text, str) else text

    probs = model.predict_proba(text_list)
    pred = np.argmax(probs, axis=1)

    return encoder.inverse_transform(pred)[0]

