import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIG
# =========================
CSV_PATH = "train.csv"
IMG_DIR = "images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

images = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, f"{row['id']}.jpg")
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img)
        img = preprocess_input(img)  # ✅ CRITICAL FIX
        images.append(img)
        labels.append(row["species"])

X = np.array(images)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)
y_cat = tf.keras.utils.to_categorical(y, num_classes)

# Save classes (USED IN STREAMLIT)
np.save("label_classes.npy", le.classes_)

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# HANDLE CLASS IMBALANCE
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))

# =========================
# DATA AUGMENTATION (HUGE BOOST)
# =========================
datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(X_train)

# =========================
# BASE MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# =========================
# CUSTOM HEAD
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
    ]
)

# =========================
# CALLBACKS
# =========================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    )
]

# =========================
# TRAIN
# =========================
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# SAVE MODEL
# =========================
model.save("cnn_leaf_model.h5")

print("✅ Improved CNN model saved as cnn_leaf_model.h5")
