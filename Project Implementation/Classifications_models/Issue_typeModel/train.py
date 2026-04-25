import pickle
from model_builder import build_model
from load_data import load_data
from config import MODEL_PATH, ENCODER_PATH, DATA_PATH
from sklearn.preprocessing import LabelEncoder


def train_model():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATA_PATH)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    num_classes = len(set(y_train_encoded))

    model = build_model(num_classes)
    model.fit(X_train, y_train_encoded)

    print("\n" + "="*45)
    print("             MODEL EVALUATION")
    print("="*45)
    
    train_acc = model.score(X_train, y_train_encoded)
    val_acc = model.score(X_val, y_val_encoded)
    test_acc = model.score(X_test, y_test_encoded)
    
    print(f"1. Training Accuracy:   {train_acc:.4f}")
    print(f"2. Validation Accuracy: {val_acc:.4f}")
    print(f"3. Test Accuracy:       {test_acc:.4f}\n")

    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
 
    return model


if __name__ == "__main__":
    train_model()

#           For final version data
#               8k rows
# =============================================
#              MODEL EVALUATION
# =============================================
# 1. Training Accuracy:   0.9498
# 2. Validation Accuracy: 0.8805
# 3. Test Accuracy:       0.8774



#             5k rows
# =============================================
#              MODEL EVALUATION
# =============================================
# 1. Training Accuracy:   0.9591
# 2. Validation Accuracy: 0.9205
# 3. Test Accuracy:       0.9188



