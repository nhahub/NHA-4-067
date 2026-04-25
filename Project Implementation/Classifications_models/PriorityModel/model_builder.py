import tensorflow as tf
import tensorflow_hub as hub
from config import HUB_URL

def build_model(num_classes):
    hub_layer = hub.KerasLayer(HUB_URL,
                           input_shape=[], dtype=tf.string, trainable=True)
    
    model = tf.keras.Sequential([
        hub_layer,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model