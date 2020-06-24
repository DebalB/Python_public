# Steganography: Hiding an image inside another

## What is steganography?

> [Steganography](https://en.wikipedia.org/wiki/Steganography) is the practice of concealing a file, message, image, or video within another file, message, image, or video.

## NOTE
This is an OpenCV implementation of original work done by Kelvin Salton do Prado.
Refer to the [original implementation using PIL library]( https://github.com/kelvins/steganography).
Refer to this [excellent blog post](https://towardsdatascience.com/steganography-hiding-an-image-inside-another-77ca66b2acb1) explaining the concept and his implementation.

**Why did I implement it in OpenCV?**
Because I am big fan of OpenCV and love to use it for any image processing applications I develop (not that I don't appreciate how useful the PIL library is).
There is however an obvious (order of magnitude) performance impact, possibly due to the underlying numpy library. Hence I wouldn't recommend this approach for bulk image processing applications.

## Usage
Install the requirements:

```
pip install -r requirements.txt
```

Hide and Unhide your images with:

```
python steganography.py merge --img1=res/img1.jpg --img2=res/img2.jpg --output=res/output.png
python steganography.py unmerge --img=res/output.png --output=res/output2.png
```
