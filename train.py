import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------------
# 1. LOAD DATASET FROM FOLDERS
# --------------------------------------------------------
train_dir = "Data"   # Your Data folder inside PlantSnap

datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2  # 80% training, 20% validation
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Number of classes
num_classes = len(train_data.class_indices)
print("Classes detected:", train_data.class_indices)

# --------------------------------------------------------
# 2. CREATE MODEL (MobileNetV2 Transfer Learning)
# --------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------------
# 3. TRAIN THE MODEL
# --------------------------------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# --------------------------------------------------------
# 4. SAVE THE MODEL
# --------------------------------------------------------
model.save("plant_model.h5")
print("Model saved as plant_model.h5")
