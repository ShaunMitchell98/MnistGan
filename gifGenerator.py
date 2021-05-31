import glob2
import imageio
import tensorflow_docs.vis.embed as embed


class GifGenerator:

    @staticmethod
    def generateGif(outputFileName):

        with imageio.get_writer(outputFileName, mode='I') as writer:
            filenames = glob2.glob('image*.png')
            filenames = sorted(filenames)

            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

        embed.embed_file(outputFileName)
