import pickle
from model_builder import build_model
from load_data import load_data
from config import EPOCH, MODEL_PATH, ENCODER_PATH, DATA_PATH, BATCH_SIZE
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

def train_model():
    X_train, X_val, y_train, y_val = load_data(DATA_PATH)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    num_classes = len(set(y_train_encoded))

    model = build_model(num_classes)

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        restore_best_weights=True
    )

    print("Training Deep Learning Priority Model...")
    model.fit(
        X_train,
        y_train_encoded,
        validation_data=(X_val, y_val_encoded),
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    return model

if __name__ == "__main__":
    train_model()