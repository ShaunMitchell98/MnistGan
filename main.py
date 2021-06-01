import tensorflow as tf
from train import Trainer
import mnistImages
import modelSettings
from src import csvWriter, gifGenerator
import json

with open("config.json") as json_config_file:
    config = json.load(json_config_file)

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalise the images to [-1, 1]

modelSettings = modelSettings.ModelSettings(3, 256, 60000, 100, 16)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
    .shuffle(modelSettings.bufferSize)\
    .batch(modelSettings.batchSize)

noise = tf.random.normal([1, 100])

trainer = Trainer(config)
trainer.train(train_dataset, modelSettings.epochs, modelSettings.batchSize,
              modelSettings.noiseDim, modelSettings.numExamplesToGenerate)

trainer.restore()

mnistImages.display_image(modelSettings.epochs - 1, config)
gifGenerator.GifGenerator.generateGif('dcgan.gif')
header = ['Epochs', 'Batch Size', 'Buffer Size', 'Noise Dim', 'Num Examples To Generate']
csvWriter = csvWriter.CsvWriter(config, '../output/dcgan_stats.csv')
csvWriter.write_headers(header)
csvWriter.write_stats([modelSettings.epochs, modelSettings.batchSize, modelSettings.bufferSize, modelSettings.noiseDim, modelSettings.numExamplesToGenerate])