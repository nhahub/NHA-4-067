import os

# Get the directory where config.py is located (Issue_typeModel folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths dynamically
MODEL_PATH = os.path.join(BASE_DIR, "Issue_type.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Go up two folders to 'Project Implementation' and then into 'Data'
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Data", "Final_version.csv"))