import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pickle
from config import MODEL_PATH, ENCODER_PATH

def load_trained_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

def load_encoder():
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return encoder

def predict_priority(model, text, encoder):
    text_list = [text] if isinstance(text, str) else text
    
    probs = model.predict(text_list)
    pred = np.argmax(probs, axis=1)

    # Convert the numerical prediction back to text (e.g., 'urgent', 'low')
    return encoder.inverse_transform(pred)[0]