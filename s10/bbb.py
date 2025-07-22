# Create an ImageDataGenerator for training
train_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directory structure: data/train/class1, data/train/class2, etc.
train_generator = train_datagen.flow_from_directory(
    'data/train',               # Path to training data
    target_size=(224, 224),     # Resize all images to 224x224
    batch_size=32,
    class_mode='categorical'    # Use categorical for multi-class classification
)

# Train the model
model.fit(train_generator, epochs=5)