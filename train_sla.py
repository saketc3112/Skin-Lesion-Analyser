import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import shutil
import itertools
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(123)
tf.random.set_seed(101)

# Define dataset paths
DATASET_DIR = "../input"
METADATA_FILE = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")

# Load dataset metadata
df = pd.read_csv(METADATA_FILE)

# Define lesion types
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Map image paths and lesion types
image_paths = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(DATASET_DIR, "*", "*.jpg"))}
df['path'] = df['image_id'].map(image_paths.get)
df['cell_type'] = df['dx'].map(LESION_TYPE_DICT.get)
df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

# Data Cleaning
df['age'].fillna(df['age'].mean(), inplace=True)

# Train-validation split
df_duplicates = df.groupby('lesion_id').count()
df_duplicates = df_duplicates[df_duplicates['image_id'] == 1].reset_index()
df['duplicates'] = df['lesion_id'].apply(lambda x: 'no_duplicates' if x in list(df_duplicates['lesion_id']) else 'has_duplicates')
df_train, df_val = train_test_split(df[df['duplicates'] == 'no_duplicates'], test_size=0.17, stratify=df['dx'], random_state=101)

# Prepare image directories
BASE_DIR = "base_dir"
TRAIN_DIR, VAL_DIR = os.path.join(BASE_DIR, "train_dir"), os.path.join(BASE_DIR, "val_dir")

for directory in [BASE_DIR, TRAIN_DIR, VAL_DIR]:
    os.makedirs(directory, exist_ok=True)

for category in df['dx'].unique():
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, category), exist_ok=True)

# Copy images to directories
def copy_images(image_list, src_folder, dest_folder):
    for img in image_list:
        src = os.path.join(src_folder, img + ".jpg")
        dst = os.path.join(dest_folder, df.loc[img, 'dx'], img + ".jpg")
        if os.path.exists(src):
            shutil.copyfile(src, dst)

copy_images(df_train['image_id'], DATASET_DIR, TRAIN_DIR)
copy_images(df_val['image_id'], DATASET_DIR, VAL_DIR)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=180, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
for category in df['dx'].unique():
    path = os.path.join(TRAIN_DIR, category)
    aug_gen = datagen.flow_from_directory(path, save_to_dir=path, target_size=(224, 224), batch_size=50)
    for _ in range(10):  # Adjust based on augmentation needs
        aug_gen.next()

# Model Preparation
train_batches = datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), batch_size=10)
val_batches = datagen.flow_from_directory(VAL_DIR, target_size=(224, 224), batch_size=10)

# Load MobileNet model
base_model = tf.keras.applications.MobileNet(include_top=False, pooling='avg', input_shape=(224, 224, 3))
x = Dropout(0.25)(base_model.output)
predictions = Dense(7, activation="softmax")(x)

# Compile Model
model = Model(inputs=base_model.input, outputs=predictions)
for layer in model.layers[:-23]:  # Freeze earlier layers
    layer.trainable = False

model.compile(Adam(lr=0.01), loss="categorical_crossentropy",
              metrics=[categorical_accuracy, lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=3)])

# Define callbacks
callbacks = [
    ModelCheckpoint("model.h5", monitor="val_categorical_accuracy", save_best_only=True, mode="max"),
    ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, patience=2, verbose=1, mode="max", min_lr=1e-5)
]

# Train Model
model.fit(train_batches, validation_data=val_batches, epochs=30, verbose=1, callbacks=callbacks)

# Model Evaluation
model.load_weights("model.h5")
val_loss, val_acc, val_top_3 = model.evaluate(val_batches)
print(f"Validation Accuracy: {val_acc:.4f}, Top-3 Accuracy: {val_top_3:.4f}")

# Confusion Matrix
predictions = model.predict(val_batches)
y_pred = np.argmax(predictions, axis=1)
y_true = val_batches.classes
cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df['dx'].unique(), yticklabels=df['dx'].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=df['dx'].unique()))

# Cleanup
shutil.rmtree(BASE_DIR)
