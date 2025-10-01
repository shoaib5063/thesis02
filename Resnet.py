import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Set dataset directory (local)
data_dir = "The IQ-OTHNCCD lung cancer dataset"

# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 16

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print("Classes:", class_names)

# ---- Define ResNet50 model using Keras ----
def get_resnet50(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# ---- Training function ----
def train_model(model, train_gen, val_gen, epochs=10):
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

# ---- Evaluation function ----
def evaluate_model(model, val_gen, class_names):
    val_gen.reset()
    y_true = []
    y_pred = []
    for i in range(len(val_gen)):
        x, y = val_gen[i]
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# After training and validation, evaluate on test set
def evaluate_model_on_generator(model, generator, class_names, set_name="Test"):
    generator.reset()
    y_true = []
    y_pred = []
    for i in range(len(generator)):
        x, y = generator[i]
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    print(f"\n{set_name} Set Evaluation:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# ========== TRAINING RESNET50 (Keras) ONLY ========== 
model = get_resnet50(len(class_names))
history = train_model(model, train_gen, val_gen, epochs=25)

# ========== EVALUATION ========== 
evaluate_model(model, val_gen, class_names)
# Evaluate on test set
test_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
evaluate_model_on_generator(model, test_gen, class_names, set_name="Test")