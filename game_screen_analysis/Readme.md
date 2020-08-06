## Template Matching and Text Detection in Images

This code shows an example of template matching in images followed by detecting and extracting the text contained in the matched image region.

For the demo, I am using screenshots from a popular computer game in which players are expected to purchase different objects in the course of the game.

The objective here is to detect objects in the game screen which match the **color background** of the provided template but not necessarily the exact template.

If a match is found, the text associated with the matched object also needs to be detected.

#### Sample Input:
![](GameScreen1.jpg)

#### Template to match:
![](template1.jpg)

#### Template Match output:
![](template_match_result_05.JPG)

I am using **EAST** text detector and **Pytesseract** for text recognition.

#### Download the EAST Model for text detection

The EAST Model can be downloaded from this [dropbox link](https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1.)
Once the file has been downloaded (~85 MB), unzip it and copy the .pb model file to the working directory.

#### Install Pytesseract
To install Tesseract binaries, follow the steps mentioned in [this blog](https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/).

Now install Python bindings by running following commands:
>pip install pillow
>
>pip install pytesseract

#### Usage:
> -i GameScreen1.jpg -t template1.jpg --detector frozen_east_text_detection.pb --padding 0.4 -c 70 -m 0.8


