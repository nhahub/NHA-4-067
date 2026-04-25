import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "priority_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "priority_label_encoder.pkl")
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Data", "clean_data.csv"))

EPOCH = 5 
HUB_URL = "https://tfhub.dev/google/nnlm-en-dim128/2"
BATCH_SIZE = 32