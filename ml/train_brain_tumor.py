"""
Brain Tumor EfficientNetB3 - Download Dataset + Train + Save
4 Classes: Glioma | Meningioma | Pituitary | No Tumor
Dataset: masoudnickparvar/brain-tumor-mri-dataset (7023 images)
Expected Accuracy: 98-99%

Usage:
  # Auto-download via kagglehub (requires Kaggle account):
  python train_brain_tumor.py

  # Use manually downloaded dataset:
  python train_brain_tumor.py --data_dir "C:/Users/You/Downloads/brain-tumor-mri-dataset"
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 300          # EfficientNetB3 native size
BATCH_SIZE  = 32
EPOCHS      = 30
NUM_CLASSES = 4
CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']
SAVE_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights', 'brain_tumor_efficientnetb3.h5')
HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights', 'training_history.json')


def find_dataset_dirs(base_path):
    """Search for Training and Testing directories under base_path."""
    train_dir = None
    test_dir = None
    
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d.lower() == 'training':
                train_dir = os.path.join(root, d)
            elif d.lower() == 'testing':
                test_dir = os.path.join(root, d)
    
    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not find Training/Testing directories under: {base_path}\n"
            f"Expected structure:\n"
            f"  {base_path}/\n"
            f"    Training/\n"
            f"      glioma/\n"
            f"      meningioma/\n"
            f"      notumor/\n"
            f"      pituitary/\n"
            f"    Testing/"
        )
    
    return train_dir, test_dir


def download_dataset(data_dir=None):
    """Download or locate the brain tumor MRI dataset."""
    print("=" * 60)
    print("  Step 1: Locating Brain Tumor MRI Dataset")
    print("=" * 60)
    
    if data_dir:
        # Use manually provided path
        data_dir = os.path.expandvars(os.path.expanduser(data_dir))
        print(f"Using manually provided dataset path: {data_dir}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Provided --data_dir does not exist: {data_dir}")
    else:
        # Auto-download via kagglehub
        print("Auto-downloading via kagglehub...")
        import kagglehub
        data_dir = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        print(f"Dataset downloaded to: {data_dir}")
    
    train_dir, test_dir = find_dataset_dirs(data_dir)
    
    print(f"   Training dir: {train_dir}")
    print(f"   Testing dir:  {test_dir}")
    
    # Count images per class
    for split_name, split_dir in [("Training", train_dir), ("Testing", test_dir)]:
        print(f"\n   {split_name}:")
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"     {cls}: {count} images")
    
    return train_dir, test_dir


def create_data_generators(train_dir, test_dir):
    """Create training and validation data generators with augmentation."""
    print("\n" + "=" * 60)
    print("  Step 2: Creating Data Generators")
    print("=" * 60)
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.15  # Use 15% of training data for validation
    )
    
    # Validation/Test generator (no augmentation)
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Training samples:   {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples:       {test_generator.samples}")
    print(f"   Classes: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator


def build_model():
    """Build EfficientNetB3 transfer learning model."""
    print("\n" + "=" * 60)
    print("  Step 3: Building EfficientNetB3 Model")
    print("=" * 60)
    
    # Load EfficientNetB3 with ImageNet weights, without top
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base layers initially
    base_model.trainable = False
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built successfully")
    print(f"   Total params:     {model.count_params():,}")
    print(f"   Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"   Frozen params:    {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}")
    
    return model, base_model


def train_phase1(model, train_gen, val_gen):
    """Phase 1: Train classification head only (base frozen)."""
    print("\n" + "=" * 60)
    print("  Step 4: Phase 1 Training (Frozen Base)")
    print("=" * 60)
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Phase 1 complete")
    print(f"   Best val_accuracy: {max(history1.history['val_accuracy']):.4f}")
    
    return history1


def train_phase2(model, base_model, train_gen, val_gen):
    """Phase 2: Fine-tune top 20 layers with lower learning rate."""
    print("\n" + "=" * 60)
    print("  Step 5: Phase 2 Fine-Tuning (Top 20 Layers)")
    print("=" * 60)
    
    # Unfreeze top 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"   Trainable params after unfreeze: {trainable_count:,}")
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
        ModelCheckpoint(
            SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Phase 2 complete")
    print(f"   Best val_accuracy: {max(history2.history['val_accuracy']):.4f}")
    
    return history2


def evaluate_model(model, test_gen):
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print("  Step 6: Evaluating on Test Set")
    print("=" * 60)
    
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\n✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Test Loss:     {test_loss:.4f}")
    
    return test_loss, test_acc


def save_history(history1, history2, test_acc):
    """Save training history to JSON."""
    combined = {
        'phase1': {
            'accuracy': [float(x) for x in history1.history['accuracy']],
            'val_accuracy': [float(x) for x in history1.history['val_accuracy']],
            'loss': [float(x) for x in history1.history['loss']],
            'val_loss': [float(x) for x in history1.history['val_loss']],
        },
        'phase2': {
            'accuracy': [float(x) for x in history2.history['accuracy']],
            'val_accuracy': [float(x) for x in history2.history['val_accuracy']],
            'loss': [float(x) for x in history2.history['loss']],
            'val_loss': [float(x) for x in history2.history['val_loss']],
        },
        'test_accuracy': float(test_acc),
        'classes': CLASSES,
        'model': 'EfficientNetB3',
        'img_size': IMG_SIZE,
    }
    
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"✅ Training history saved to: {HISTORY_PATH}")


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNetB3 Brain Tumor Classifier')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Path to manually downloaded dataset folder (containing Training/ and Testing/ subdirs). '
             'If not provided, kagglehub will auto-download the dataset.'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  Brain Tumor EfficientNetB3 Training Pipeline")
    print("  4 Classes: Glioma | Meningioma | Pituitary | No Tumor")
    print("=" * 60 + "\n")
    
    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Step 1: Download / locate dataset
    train_dir, test_dir = download_dataset(args.data_dir)
    
    # Step 2: Create data generators
    train_gen, val_gen, test_gen = create_data_generators(train_dir, test_dir)
    
    # Step 3: Build model
    model, base_model = build_model()
    
    # Step 4: Phase 1 - Train classification head
    history1 = train_phase1(model, train_gen, val_gen)
    
    # Step 5: Phase 2 - Fine-tune top layers
    history2 = train_phase2(model, base_model, train_gen, val_gen)
    
    # Step 6: Evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_gen)
    
    # Step 7: Save model and history
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    model.save(SAVE_PATH)
    print(f"\n✅ Model saved to: {SAVE_PATH}")
    
    save_history(history1, history2, test_acc)
    
    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print(f"  Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Model: {SAVE_PATH}")
    print(f"  History: {HISTORY_PATH}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
