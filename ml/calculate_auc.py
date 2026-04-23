import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score

# Paths
model_path = os.path.join('models', 'weights', 'brain_tumor_efficientnetb3.h5')
test_dir = os.path.join('data', 'brain-tumor-dataset', 'Testing')
IMG_SIZE = 300
BATCH_SIZE = 32

print("Loading model...")
model = tf.keras.models.load_model(model_path)

print("Preparing test data...")
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Predicting on test set...")
predictions = model.predict(test_generator, verbose=1)

print("Calculating AUC...")
true_labels = test_generator.classes
# Calculate one-vs-rest AUC score
auc = roc_auc_score(true_labels, predictions, multi_class='ovr')

print("=" * 40)
print(f"✅ Overall AUC Score (OVR): {auc:.4f}")
print("=" * 40)
