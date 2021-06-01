import PIL
import matplotlib.pyplot as plt
import os


def generate_and_save_images(model, settings, epoch, test_input):
    # Notice 'training' is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(settings.Output, 'image_at_epoch_{:04d}.png'.format(epoch+1)))


def display_image(epoch_no, settings):
    return PIL.Image.open(os.path.join(settings.Output, 'image_at_epoch_{:04d}.png'.format(epoch_no)))
