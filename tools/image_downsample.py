import os
from os.path import join
from glob import glob
from PIL import Image
from tqdm import tqdm


def downsample_image(image, factor):
    """
    Downsample an image by a factor of 2.
    """
    return image.resize((image.size[0]//factor, image.size[1]//factor))


def downsample_images(path, foloder, factor):
    """
    Downsample all images in a directory.
    """
    

    extensions = ['.jpg', '.png', '.JPG', '.PNG']
    images = sorted(sum([
        glob(join(path, foloder, 'video', '*'+ext)) for ext in extensions
        ], []))
    for image in tqdm(images):
        im = Image.open(image)
        im = downsample_image(im, factor)
        im.save(image)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    # downsample video
    downsample_images(args.path, 'images', factor=2)

    # downsample background image
    bg = downsample_image(Image.open(join(args.path, 'bg.jpg')), 2)
    bg.save(join(args.path, 'bg.jpg'))