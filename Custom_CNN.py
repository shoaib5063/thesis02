import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action="ignore")


import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
# Use keras directly for all imports
# from keras.applications.vgg19 import VGG19  # Not used in code
# from keras.optimizers import Adam  # Not used in code
# from keras.losses import SparseCategoricalCrossentropy  # Not used in code
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler

import sklearn.metrics as metrics
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
import os
import pandas as pd



# Build the dataframe from all images and labels
import os
import pandas as pd
benign_dir = 'The IQ-OTHNCCD lung cancer dataset/Bengin cases'
malignant_dir = 'The IQ-OTHNCCD lung cancer dataset/Malignant cases'
normal_dir = 'The IQ-OTHNCCD lung cancer dataset/Normal cases'

filepaths = []
labels = []
for f in os.listdir(benign_dir):
    filepaths.append(os.path.join(benign_dir, f))
    labels.append('cancer')
for f in os.listdir(malignant_dir):
    filepaths.append(os.path.join(malignant_dir, f))
    labels.append('cancer')
for f in os.listdir(normal_dir):
    filepaths.append(os.path.join(normal_dir, f))
    labels.append('no_cancer')

Lung_df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
print(Lung_df['labels'].value_counts())
print(Lung_df.head())

# Split into train, validation, and test sets
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(Lung_df, test_size=0.2, stratify=Lung_df['labels'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['labels'], random_state=42)
print('Train:', train_df.shape)
print('Validation:', val_df.shape)
print('Test:', test_df.shape)

# Image generators
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rescale=1./255)
train_gen = image_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepaths",
    y_col="labels",
    target_size=(224,224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=4,
    shuffle=True
)
val_gen = image_gen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepaths",
    y_col="labels",
    target_size=(224,224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=4,
    shuffle=False
)
test_gen = image_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepaths",
    y_col="labels",
    target_size=(224,224),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=4,
    shuffle=False
)
classes = list(train_gen.class_indices.keys())
print('Classes:', classes)

# Optional: visualize a batch
import matplotlib.pyplot as plt
import numpy as np
def show_images(generator):
    images, labels = next(generator)
    plt.figure(figsize=(12,8))
    for i in range(min(8, len(images))):
        plt.subplot(2,4,i+1)
        plt.imshow(images[i])
        class_name = classes[np.argmax(labels[i])]
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
show_images(train_gen)

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(8, 8), strides=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3)),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    #keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    #keras.layers.BatchNormalization(),
    #keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()
from keras.utils import plot_model

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
## Removed duplicate import, already imported from keras.callbacks

early_stopping = EarlyStopping(
    monitor='val_loss',       # or 'val_accuracy'
    patience=5,               # stop if no improvement for 5 epochs
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate on validation set
val_preds = model.predict(val_gen, verbose=1)
val_true = np.argmax(np.concatenate([y for x, y in val_gen], axis=0), axis=1)
val_pred = np.argmax(val_preds, axis=1)
print("\nValidation Set Evaluation:")
print(classification_report(val_true, val_pred, target_names=classes))
print("Confusion Matrix:\n", confusion_matrix(val_true, val_pred))

# Evaluate on test set
test_preds = model.predict(test_gen, verbose=1)
test_true = np.argmax(np.concatenate([y for x, y in test_gen], axis=0), axis=1)
test_pred = np.argmax(test_preds, axis=1)
print("\nTest Set Evaluation:")
print(classification_report(test_true, test_pred, target_names=classes))
print("Confusion Matrix:\n", confusion_matrix(test_true, test_pred))