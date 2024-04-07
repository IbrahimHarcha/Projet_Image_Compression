from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Définition des paramètres
img_height = 150
img_width = 150
batch_size = 32
epochs = 10

# Créer des générateurs pour les données d'entraînement et de validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train_data_directory',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'validation_data_directory',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)
