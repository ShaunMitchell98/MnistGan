import glob2
import imageio
import os
import tensorflow_docs.vis.embed as embed


class GifGenerator:

    @staticmethod
    def generateGif(settings, outputFileName):

        file_path = os.path.join(settings.Output, outputFileName)
        with imageio.get_writer(file_path, mode='I') as writer:
            filenames = glob2.glob(os.path.join(settings.Output, 'image*.png'))
            filenames = sorted(filenames)

            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

        embed.embed_file(file_path)
