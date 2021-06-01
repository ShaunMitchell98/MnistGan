import tensorflow as tf
from train import Trainer
from src import csvWriter, gifGenerator, mnistImages, modelSettings
import json
import deserialize
import emptyDirectory

with open("../config.json") as json_config_file:
    config = json.load(json_config_file)

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalise the images to [-1, 1]

settings = deserialize.deserialize(modelSettings.ModelSettings(), config)

emptyDirectory.emptyDirectory(settings.Output)
emptyDirectory.emptyDirectory(settings.TrainingCheckpoints)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
    .shuffle(settings.BufferSize)\
    .batch(settings.BatchSize)

noise = tf.random.normal([1, 100])

trainer = Trainer(settings)
trainer.train(train_dataset, settings.Epochs, settings.BatchSize,
              settings.NoiseDim, settings.NumExamplesToGenerate)

trainer.restore()

mnistImages.display_image(settings.Epochs - 1, settings)
gifGenerator.GifGenerator.generateGif(settings, 'dcgan.gif')
