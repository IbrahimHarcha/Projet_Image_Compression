# Importation des générateurs de données et du modèle
from data_preprocessing import train_generator, validation_generator
from model_construction import model

# Définition des paramètres
epochs = 10

# Entraînement du modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size
)
