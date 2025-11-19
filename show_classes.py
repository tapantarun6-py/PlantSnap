from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "Data",
    target_size=(224,224),
    batch_size=32,
    subset="training"
)

print("\nCLASS ORDER:")
print(train_data.class_indices)
