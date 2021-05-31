import tensorflow as tf
import generator
import discriminator
import mnistImages
import os
import csvWriter
import numpy as np

from IPython import display

import time


class Trainer:

    def __init__(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.generatorModel = generator.Generator.make_model()
        self.discriminatorModel = discriminator.Discriminator.make_model()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generatorModel,
                                              discriminator=self.discriminatorModel)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.generatorCsvWriter = csvWriter.CsvWriter('Generator Training Losses.csv')
        self.generatorCsvWriter.write_headers(['Epoch', 'Average Loss'])
        self.discriminatorCsvWriter = csvWriter.CsvWriter('Discriminator Training Losses.csv')
        self.discriminatorCsvWriter.write_headers(['Epoch', 'Average Loss'])

    @tf.function
    def train_step(self, images, batchSize, noiseDim):
        noise = tf.random.normal([batchSize, noiseDim])
        gen_loss = 0
        disc_loss = 0

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generatorModel(noise, training=True)

            real_output = self.discriminatorModel(images, training=True)
            fake_output = self.discriminatorModel(generated_images, training=True)

            gen_loss = generator.Generator().loss(fake_output)
            disc_loss = discriminator.Discriminator().loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generatorModel.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminatorModel.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generatorModel.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            (zip(gradients_of_discriminator, self.discriminatorModel.trainable_variables)))
        return [gen_loss, disc_loss]

    def train(self, dataset, epochs, batchSize, noiseDim, numExamplesToGenerate):

        # You will reuse this seed over time (so it's easier)
        # to visualise progress in the animated GIF)
        seed = tf.random.normal([numExamplesToGenerate, noiseDim])

        for epoch in range(epochs):
            start = time.time()
            epoch_generator_losses = []
            epoch_discriminator_losses = []

            for image_batch in dataset:
                losses = self.train_step(image_batch, batchSize, noiseDim)
                epoch_generator_losses.append(losses[0])
                epoch_discriminator_losses.append(losses[1])

                # Save the model every 15 epochs
                if (epoch + 1) % 15 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            display.clear_output(wait=True)
            mnistImages.generate_and_save_images(self.generatorModel, epoch, seed, False)
            self.generatorCsvWriter.write_stats([epoch, np.average(epoch_generator_losses)])
            self.discriminatorCsvWriter.write_stats([epoch, np.average(epoch_discriminator_losses)])

        # Generate after the final epoch
        display.clear_output(wait=True)
        mnistImages.generate_and_save_images(self.generatorModel, epoch, seed, True)

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
