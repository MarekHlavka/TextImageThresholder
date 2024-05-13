# ZPO Project - Text Thresholding - Marek Hlávka (xhlavk09)

## Introduction

The aim of the project is to perform text thresholding on images to separate text from the background (assuming dark text on a light background) or alternatively,
for example, sketches from the background. Before actual thresholding, perform an automatic analysis of the image histogram and select the most suitable threshold
(the histogram should have two very prominent peaks). Experiment with various types of thresholding algorithms and histogram analysis.
Document the procedures and evaluate them in terms of accuracy and speed.

## Project Goals

The main goal of the project is to develop a program that can, at least in terms of time, automatically compare thresholding algorithms and objectively
evaluate the best among them. In the entire process of processing images with text, this step (quality thresholding) is very important for several reasons.
Firstly, text is much better recognized from binary images, but several factors can significantly affect the resulting binary image. These factors include non-uniform
illumination across the image or non-uniform strength and quality of the text. As a result, errors such as skipping text or mistaking a piece of background for text
may occur during the text recognition step. Quality thresholding can eliminate these problems, especially focusing on eliminating the problem of non-uniform illumination,
as it is a common issue when capturing images of handwritten text, especially without access to a scanner (e.g., when taking photos with a phone).

## Usage

Program has many arguments and several libraries and modules are needed for its successfull run.
Directory contains file 'requirements.txt', that can be used with progem 'pip' and prepare enviroment
for this rpogram:
    $ pip install -r requirements.txt
Program can be then executed with following command with Python:
    $ python/python3 zpo.py arguments...

Usage: zpo.py [-h] [-s source_dir] [-v valid_dir] [-a valid_append] [-e extension] [-p N N]
[--graphs | --no-graphs] [--limit N of iterations] [--save save]
ZPO project
options:
-h, --help                              show this help message and exit
-s source_dir, --source source_dir      Source directory for input images
-v valid_dir, --valid valid_dir         Directory with ground truth/valid images
-a valid_append, --append valid_append  Appen for valid filename filenames  Append = '_valid', Input file:'infile.png' then is expected valid file 'infile_valid.png'
-e extension, --extension extension     Extensions of input and valid files
-p N N, --parts N N                     Rows and Cols to divide image with tiled thresholing, if not present thresolding is done globally
--graphs, --no-graphs                   If there should be matrix of binary image for each image and method after each image processing shown
--limit N of iterations                 Limit number of loaded files
--save save                             Flag if saving mages should be done, and aslo index of method of thresholding
                        [0] - Otsu method
                        [1] - Midvalue method
                        [2] - adaptive_mean_50
                        [3] - adaptive_mean_100
                        [4] - adaptive_gauss_50
                        [5] - adaptive_gauss_100
                        [6] - RiddlerCalvard method
                        [7] - Two Gausses interpolation
                        [8] - Peak and valleys method

## Test Images

For this project, test images were selected from two datasets - Grantha binarization dataset and Captcha Images. Images from these
two datasets were selected because they meet the conditions of input images specified in the project requirements.

## Project Concept and Program Design

The project is developed in Python with significant support from the OpenCV library. Compared to C++, Python was chosen primarily due to
considering the implementation of a simple optional GUI for quickly displaying all results at once or for the user to easily choose the threshold value.
The main way of using the program is to threshold the provided image using the chosen method, all methods at once, or the same but with a sequence of images.
In addition to the thresholded images themselves, the program provides statistics regarding the time complexity of individual algorithms. However,
evaluating the accuracy of algorithms is left to the subsequent examination of the created images by the user. This is because, until the current moment,
images with high-quality reference variants for thresholding have not been found.

### Selected Algorithms for Text Thresholding

   - Mid-value Thresholding between Two Histogram Peaks
   - Otsu's Thresholding Method
   - Fitting Normal Distributions
   - Riddler-Calvard Method
   - Selection of the Best Minimum
   - Standard Adaptive Methods

In addition to traditional adaptive methods, the script will combine the above-mentioned global thresholding methods and an adaptive approach.
It will divide the input image into smaller tiles and threshold them separately for each algorithm. The resulting binary images will then be
merged back into a whole image, which will be the resulting thresholded image.

### Description of Selected Algorithms for Image/Text Thresholding

   - Mid-value Thresholding between Two Histogram Peaks - This method is straightforward and serves more as a comparison to consider how complex
   a method must be to be effective. It relies on histogram analysis and finding two values that occur most frequently in the image. From these
   values, an average is calculated and used as the threshold for the entire image (or part, in the case of adaptation to local thresholding).

   - Otsu's Thresholding Method - Otsu's method is a popular algorithm for automatically choosing a threshold value for image binarization.
   It aims to find the optimal threshold value to minimize the weighted sum of intra-class variances of two classes of pixels in the image (background and foreground).
   Otsu's method is based on the analysis of the image histogram and calculation of a threshold value that maximizes the separation between the two classes of pixels.
   This approach allows for effective image binarization even in cases where the histogram is poorly defined or has multiple peaks.

   - Fitting Normal Distributions - This method again relies on histogram analysis. It is based on fitting the histogram with two curves representing normal
   distributions (assuming the histogram has two prominent peaks), joining them, and then finding the local minimum between their peaks. The value where this
   minimum is located is then used as the threshold value.

   - Selection of the Best Minimum - This method works with local minima of the histogram. All values with minimum occurrence (local minima) are selected,
   and one minimum is chosen based on statistical methods, and the value representing this minimum is selected as the threshold value.

   - Riddler-Calvard Method - This method is used for automatically selecting a threshold value for image binarization. This algorithm aims to find the
   optimal threshold value to maximize the separation between two classes of pixels in the image, typically background and foreground. The Riddler-Calvard
   method iteratively calculates a threshold value by taking the average of the pixel intensities of the foreground and background regions,
   until the threshold value converges to a stable value.

   - Standard Adaptive Methods - Common adaptive thresholding methods such as Gaussian adaptive thresholding, mean adaptive thresholding, etc., will also be implemented for comparison.

## Results

The program successfully processes the entire set of images using all selected methods. The main limitation is the high processing time due to the large
number of thresholding methods implemented. When using a sequence of images for processing, the average time is increased even more. However, it is necessary
to consider that this is a prototype of the final program and that the time complexity could be further reduced, for example, by parallel processing of images,
by using a better-suited data structure, or by rewriting critical parts in a compiled language such as C++.