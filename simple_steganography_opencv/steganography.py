#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Usage
# python simple_steganography_opencv.py merge --img1=res/img1.jpg --img2=res/img2.jpg --output=res/output.png
# python simple_steganography_opencv.py unmerge --img=res/output.png --output=res/output2.png

import click
import cv2
import numpy as np
import time
import imutils

class Steganography(object):

    @staticmethod
    def __int_to_bin(rgb):
        """Convert an integer tuple to a binary (string) tuple.

        :param rgb: An integer tuple (e.g. (220, 110, 96))
        :return: A string tuple (e.g. ("00101010", "11101011", "00010110"))
        """
        r, g, b = rgb
        return ('{0:08b}'.format(r),
                '{0:08b}'.format(g),
                '{0:08b}'.format(b))

    @staticmethod
    def __bin_to_int(rgb):
        """Convert a binary (string) tuple to an integer tuple.

        :param rgb: A string tuple (e.g. ("00101010", "11101011", "00010110"))
        :return: Return an int tuple (e.g. (220, 110, 96))
        """
        r, g, b = rgb
        return (int(r, 2),
                int(g, 2),
                int(b, 2))

    @staticmethod
    def __merge_rgb(rgb1, rgb2):
        """Merge two RGB tuples.

        :param rgb1: A string tuple (e.g. ("00101010", "11101011", "00010110"))
        :param rgb2: Another string tuple
        (e.g. ("00101010", "11101011", "00010110"))
        :return: An integer tuple with the two RGB values merged.
        """
        r1, g1, b1 = rgb1
        r2, g2, b2 = rgb2
        rgb = (r1[:4] + r2[:4],
               g1[:4] + g2[:4],
               b1[:4] + b2[:4])
        return rgb

    @staticmethod
    def merge(img1, img2):
        """Merge two images. The second one will be merged into the first one.

        :param img1: First image
        :param img2: Second image
        :return: A new merged image.
        """
        
        new_image = []
        pixel_map1 = []
        pixel_map2 = []
        
        if type(img1) == 'numpy.ndarray':
          im1_shape = img1.shape
          new_image = img1.copy()
          pixel_map1 = img1
        else:
          im1_shape = img1.get().shape
          new_image = img1.get().copy()
          pixel_map1 = img1.get()

        if type(img2) == 'numpy.ndarray':
          im2_shape = img2.shape
          pixel_map2 = img2
        else:
          im2_shape = img2.get().shape
          pixel_map2 = img2.get()

        # Check the images dimensions
        # if img2.shape[0] > img1.shape[0] or img2.shape[1] > img1.shape[1]:
        if im2_shape[0] > im1_shape[0] or im2_shape[1] > im1_shape[1]:
            raise ValueError('Image 2 should not be larger than Image 1!')

        # Get the pixel map of the two images
        
        

        # Create a new image that will be outputted
        # new_image = img1.copy()
        pixels_new = new_image

        for i in range(im1_shape[0]):
            for j in range(im1_shape[1]):
                rgb1 = Steganography.__int_to_bin(pixel_map1[i, j])

                # Use a black pixel as default
                rgb2 = Steganography.__int_to_bin((0, 0, 0))

                # Check if the pixel map position is valid for the second image
                if i < im2_shape[0] and j < im2_shape[1]:
                    rgb2 = Steganography.__int_to_bin(pixel_map2[i, j])

                # Merge the two pixels and convert it to a integer tuple
                rgb = Steganography.__merge_rgb(rgb1, rgb2)

                pixels_new[i, j] = Steganography.__bin_to_int(rgb)

        return new_image

    @staticmethod
    def unmerge(img):
        """Unmerge an image.

        :param img: The input image.
        :return: The unmerged/extracted image.
        """

        # Load the pixel map
        pixel_map = img

        # Create the new image and load the pixel map
        new_image = img.copy()
        pixels_new = new_image
        
        # Tuple used to store the image original size
        original_size = img.shape
        h = original_size[0]
        w = original_size[1]
        
        # cond_time = 0

        # start_time = time.time()
        for i in range(h):
            for j in range(w):
                # Get the RGB (as a string tuple) from the current pixel
                r, g, b = Steganography.__int_to_bin(pixel_map[i, j])

                # Extract the last 4 bits (corresponding to the hidden image)
                # Concatenate 4 zero bits because we are working with 8 bit
                rgb = (r[4:] + '0000',
                       g[4:] + '0000',
                       b[4:] + '0000')

                # Convert it to an integer tuple
                pixels_new[i, j] = Steganography.__bin_to_int(rgb)

                # If this is a 'valid' position, store it
                # as the last valid position
                # cond_start = time.time()
                if tuple(pixels_new[i, j]) != (0,0,0):
                    original_size = (i + 1, j + 1)
                # cond_end = time.time()
                # cond_time +=  (cond_end-cond_start)
                    
        # end_time = time.time()
        # print("Total time taken for condition={:0.02f}s".format(cond_time))
        # print("Total time taken for loop={:0.02f}s".format(end_time-start_time))
        
        # Crop the image based on the 'valid' pixels
        new_image = new_image[:original_size[0],:original_size[1]]

        return new_image


@click.group()
def cli():
    pass


@cli.command()
@click.option('--img1', required=True, type=str, help='Image that will hide another image')
@click.option('--img2', required=True, type=str, help='Image that will be hidden')
@click.option('--output', required=True, type=str, help='Output image')
def merge(img1, img2, output):
    start_time = time.time()
    # merged_image = Steganography.merge(cv2.imread(img1), cv2.imread(img2))
    merged_image = Steganography.merge(cv2.UMat(cv2.imread(img1)), cv2.UMat(cv2.imread(img2)))
    end_time = time.time()
    print("Total time taken for merge={:0.02f}".format(end_time-start_time))
    cv2.imwrite(output,merged_image)
    cv2.imshow("Input Image-1", imutils.resize(cv2.imread(img1),width=1024))
    cv2.imshow("Input Image-2", imutils.resize(cv2.imread(img2),width=1024))
    cv2.imshow("Merged Image", imutils.resize(cv2.imread(output),width=1024))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@cli.command()
@click.option('--img', required=True, type=str, help='Image that hides another image')
@click.option('--output', required=True, type=str, help='Output image')
def unmerge(img, output):
    start_time = time.time()
    unmerged_image = Steganography.unmerge(cv2.imread(img))
    end_time = time.time()
    print("Total time taken for unmerge={:0.02f}s".format(end_time-start_time))
    cv2.imwrite(output,unmerged_image)
    cv2.imshow("Original Image", imutils.resize(cv2.imread(img),width=1024))
    cv2.imshow("Unmerged Image", imutils.resize(cv2.imread(output),width=1024))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cli()
