import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    X = df['initial_message']
    y = df['priority']

    return train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)