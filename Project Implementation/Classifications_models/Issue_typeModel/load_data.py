import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    X = df['initial_message']
    y = df['refined_issue']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test