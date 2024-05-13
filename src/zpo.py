import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from scipy.signal import find_peaks
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter1d
from argparse import ArgumentParser, BooleanOptionalAction, RawTextHelpFormatter
import mahotas

def tear_down():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def args_handle():
    parser = ArgumentParser(description ='ZPO project', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', '--source', metavar='source_dir', type=str, nargs=1, help="Source directory for input images")
    parser.add_argument('-v', '--valid', metavar='valid_dir', type=str, nargs=1, help="Directory with ground truth/valid images")
    parser.add_argument('-a', '--append', metavar='valid_append', type=str, nargs=1, help="""Appen for valid filename filenames
        Append = '_valid', Input file:'infile.png' then is expected valid file 'infile_valid.png'""")
    parser.add_argument('-e', '--extension', metavar='extension', type=str, nargs=1, help="Extensions of input and valid files")
    parser.add_argument('-p', '--parts', metavar='N', type=int, nargs=2, help="Rows and Cols to divide image with tiled thresholing, if not present thresolding is done globally")
    parser.add_argument('--graphs', action=BooleanOptionalAction, help="If there should be matrix of binary image for each image and method after each image processing shown")
    parser.add_argument('--limit', metavar='N of iterations', type=int, nargs=1, help="Limit number of loaded files")
    parser.add_argument('--save', metavar='save', type=int, nargs=1, help="""Flag if saving mages should be done, and aslo index of method of thresholding
        [0] - Otsu method
        [1] - Midvalue method
        [2] - adaptive_mean_50
        [3] - adaptive_mean_100
        [4] - adaptive_gauss_50
        [5] - adaptive_gauss_100
        [6] - RiddlerCalvard method
        [7] - Two Gausses interpolation
        [8] - Peak and valleys method""")
    return parser.parse_args()

# Class that contains all methods for thresholfing in this project
class ThreshMethods:
    def __init__(self):
        self.methods = [
            self.otsu_thresold,
            self.value_middle_from_two_maxvals,
            self.adaptive_mean_50,
            self.adaptive_mean_100,
            self.adaptive_gauss_50,
            self.adaptive_gauss_100,
            self.riddler_calvard,
            self.two_gausses,
            self.peak_and_valleys_method
        ]

    # Method for running all methods in sequence on one image
    # 'img' - input image
    # 'save' - flag if output will be saved
    # return - list of processed images and list of run times of those process times
    def run_methods(self, img, save):
        ret_images = []
        run_times = []

        # Saving mathodname to add it to filename in output
        if save is not False:
            new_res = {
                "name": self.methods[save].__name__,
                "image": self.methods[save](img)
            }

            ret_images.append(new_res)
            return ret_images

        # Running all methods
        for method in self.methods:
            
            start_time = time.time()
            new_res = {
                "name": method.__name__,
                "image": method(img)
            }
            end_time = time.time()

            ret_images.append(new_res)
            run_times.append((end_time - start_time) * 1000.0)

        return ret_images, run_times


    # binary thresholding for mathods that only calculates thresholding value
    #   and not performing thresholding itself
    @staticmethod
    def bin_thresold(img, val: int):
        ret, binary_image = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
        return binary_image

    def otsu_thresold(self, img):
        ret, binary_image = cv2.threshold(img, None, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary_image
    
    def value_middle_from_two_maxvals(self, img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        
        # Get maximum values in histogram (but with some distance - not neighbor values)
        loc_maximas, peak_data = find_peaks(hist.flatten(), distance=(hist.flatten().size/10.0), height=0.1)
        
        highest_peaks = []
        highest_peaks.append(loc_maximas[np.argmax(peak_data['peak_heights'])])                 # Max value
        highest_peaks.append(loc_maximas[np.argpartition(peak_data['peak_heights'],-2)[-2]])    # Second max value

        middle_value = int((highest_peaks[0] + highest_peaks[1])/2)
        return ThreshMethods.bin_thresold(img, middle_value)
    
    def adaptive_mean_50(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 49, 5) 
    def adaptive_mean_100(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 99, 5) 
    def adaptive_gauss_50(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 49, 5)
    def adaptive_gauss_100(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 99, 5)
    
    # Riddler calvard method
    def riddler_calvard(self, img):
        T = mahotas.thresholding.rc(img)
        return ThreshMethods.bin_thresold(img, T)
    
    # Return double gaussian function
    @staticmethod
    def double_gaussian(x, params):
            (c1, mu1, sigma1, c2, mu2, sigma2) = params
            res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
                  + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
            return res
    
    # Get fitting function as diff
    @staticmethod
    def double_gaussian_fit(params, x, y):
            fit = ThreshMethods.double_gaussian(x, params)
            return (fit - y)

    # Perform gaussian fiting by iterating and calculating diff via least square method
    def two_gausses(self, img):
        data = cv2.calcHist([img], [0], None, [256], [0, 256])

        y = data[:, 0]
        x = [ i for i in range(0, len(y))]

        # Remove background.
        y_proc = np.copy(y)
        y_proc[y_proc < 5] = 0.0

        # Least squares fit. Starting values found by inspection.
        fit = leastsq(ThreshMethods.double_gaussian_fit, [80.0,50.0,5.0,  80.0,200.0,5.0], args=(x, y_proc))

        # Get mean and standart deviation
        first = {
            'mag': fit[0][0],
            'pos': fit[0][1]
        }
        second = {
            'mag': fit[0][3],
            'pos': fit[0][4]
        }

        # Bigger the magnitude, closer the threshold is to the mean of gauss line
        ratio = first['mag'] / (first['mag']+second['mag'])
        space_between = second['pos'] - first['pos']
        threshold = first['pos'] + (space_between*ratio)

        return ThreshMethods.bin_thresold(img, threshold)
    
    def g_2(self, x):
        return 0 if x == 0 else 1 if x > 0 else -1

    # Return index of valley with minimal value 'abs(data[valley_indexes[i]] - avg_valley + pow(derivation[i],2))'
    def get_minimal_valley(self, valley_indexes, data, derivation):
        avg_valley = np.average([data[i] for i in valley_indexes])

        min_index = None
        min_value = sys.float_info.max

        for i in range(0, len(valley_indexes)):
            val = abs(data[valley_indexes[i]] - avg_valley + pow(derivation[i],2))
            if val <= min_value:
                min_value = val
                min_index = valley_indexes[i]

        return min_index


    # Method for selecting thresholfing method as best valley index
    def peak_and_valleys_method(self, img):

        data = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        derivation = np.gradient(data)
        sign_change = [self.g_2(x) for x in derivation]
        
        valley_indexes = []

        for i in range(0, len(sign_change) - 1):
            if sign_change[i] < sign_change[i+1]:
                valley_indexes.append(i)

        threshold_value = self.get_minimal_valley(valley_indexes, data, derivation)
        return ThreshMethods.bin_thresold(img, threshold_value)

# ------------------------------------------------------------------------
# Functions for calculating differences
def img_diff(img1, img2):
    res = cv2.absdiff(img1, img2)
    res = res.astype(np.uint8)
    percentage = 100.0 - (np.count_nonzero(res) * 100)/ res.size
    return percentage

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                   # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
    return psnr 

# ------------------------------------------------------------------------
# Function for ploting images of each method for one in image
def plot_images(bin_images: list, src_img, dst_img, methods=None):
    side = math.ceil(math.sqrt(len(bin_images)))

    # plt.imshow(src_img, cmap='gray')

    fig = plt.figure(layout='constrained', figsize=(10, 10))
    subfigs = fig.subfigures(2, 1, hspace=0.05, height_ratios=[1, 3])

    top = subfigs[0].subplots(nrows=1, ncols=3)

    top[0].imshow(src_img, cmap='gray')
    top[0].set_title("Input")

    top[1].imshow(dst_img, cmap='gray')
    top[1].set_title("Target")

    hist = cv2.calcHist([src_img],[0],None,[256],[0,256]) 
    top[2].plot(hist)
    top[2].set_title("Histogram")

    ax = subfigs[1].subplots(nrows=side, ncols=side)
    for i in range(0, len(bin_images)):
        loc_ax = ax[math.floor(i/side), i%side]
        loc_ax.imshow(bin_images[i], cmap='gray')
        if methods:
            loc_ax.set_title(methods[i].__name__)

    plt.show()

# Simple wraping method
def process_images(input_img, methods, save):
    binary_images_results, run_times = methods.run_methods(input_img, save)

    return [img.get("image") for img in binary_images_results ], run_times

# Method for performing process image on tiles in one input image
def process_tiled_images(input_file, cols, rows, thresh_methods, save):
    #Divide source image
    nCols = math.floor(input_file.shape[1]/cols)
    nRows = math.floor(input_file.shape[0]/rows)

    rowsLim = input_file.shape[0] - nRows + (1 if (input_file.shape[0] % rows == 0) else 0)
    colsLim = input_file.shape[1] - nCols + (1 if (input_file.shape[1] % cols == 0) else 0)

    part_bin_images = []
    run_times = [0 for _ in range(0, len(thresh_methods.methods))]

    # Create output images
    for _ in range(0, len(thresh_methods.methods)):
        part_bin_images.append(input_file.copy())

    # Process each tile of input image
    for r in range(0,rowsLim, nRows):
        for c in range(0,colsLim, nCols):
            window = input_file[r:r+nRows,c:c+nCols]    # Tile

            processed_part, part_run_times = process_images(window, thresh_methods, save)
            for i in range(0, len(part_bin_images)):
                part_bin_images[i][r:r+nRows,c:c+nCols] = processed_part[i]
                run_times[i] = run_times[i] + part_run_times[i]
    
    return part_bin_images, run_times

# Calculate PSNR and OpenCV image differnce from ground truth
def calculate_diff(binary_images, target_img):
    psnr_diff_images = []
    cv_diff_images = []
    for img in binary_images:
        error = PSNR(target_img, img)
        diff = img_diff(target_img, img)

        psnr_diff_images.append(error)
        cv_diff_images.append(diff)
        
    
    return psnr_diff_images, cv_diff_images


# ------------------------------//------\\--------------------------------
# ------------------------------|| MAIN ||--------------------------------
# ------------------------------\\------//--------------------------------
if __name__ == '__main__':
    
    args = vars(args_handle())


    source_dir = "" if args.get('source') == None else args.get('source')[0]        # Source directory of input images
    valid_dir = "" if args.get('valid') == None else args.get('valid')[0]           # Directory of ground truth images
    append_valid = "" if args.get('append') == None else args.get('append')[0]      # Appended filename part of ground
                                                                                    #       truth images (e.g. "filename_valid")
    ext = "" if args.get('extension') == None else args.get('extension')[0]         # Extension of input files
    rows = 1 if args.get('parts') == None else args.get('parts')[0]                 # Count of tiles in row if used
    cols = 1 if args.get('parts') == None else args.get('parts')[1]                 # Count of tiles in cilumn if used
    tiled_processing = args.get('parts') != None                                    # Flag for tiling
    save = False if args.get('save') == None else args.get('save')[0]               # Flag if saving files or displaying statistics

    iter_graphs = args.get('graphs') != None                                        # Flag if displaing images after each input image
    iterations = None if args.get('limit') == None else args.get('limit')[0]        # Limit number of processed input files
                                                       
    thresh_methods = ThreshMethods()

    # check for valid method index
    if save:
        if save > len(thresh_methods.methods) or save < 0:
            save = 0

    # Init output lists
    methods_diffs = [[] for _ in range(0, len(thresh_methods.methods))]
    cv_methods_diffs = [[] for _ in range(0, len(thresh_methods.methods))]
    run_times = [[] for _ in range(0, len(thresh_methods.methods))]

    # Limit iterations
    img_cnt = 0

    # Load files
    for file in os.listdir(os.fsencode(source_dir)):

        # Iteration check
        if iterations != None:
            if img_cnt >= iterations:
                break
        img_cnt+=1

        # Split filename
        filename = os.path.splitext(os.fsdecode(file))[0]
        split_filename = filename.split('_', 1)[0]

        # Load input and ground truth file
        input_file = cv2.imread(f"{source_dir}/{filename}.{ext}", cv2.IMREAD_GRAYSCALE)
        target_file = cv2.imread(f"{valid_dir}/{split_filename}{append_valid}.{ext}", cv2.IMREAD_GRAYSCALE)

        # Resize input file according to output file
        input_file = cv2.resize(input_file, np.transpose(target_file).shape, 
               interpolation = cv2.INTER_CUBIC)

        
        if(tiled_processing):   # Process whole image
            binary_images = img_run_times = process_tiled_images(input_file, cols, rows, thresh_methods, save)
        else:                   # Process divided file
            binary_images, img_run_times = process_images(input_file, thresh_methods, save)

        # Calculate error in output images
        diff_images, cv_diff_images = calculate_diff(binary_images, target_file)
        
        if iter_graphs:
            plot_images(binary_images, input_file, target_file, thresh_methods.methods)
        
        # Save file if desired
        if save is not False:
            cv2.imwrite(f"{source_dir}/{filename}_{thresh_methods.methods[save].__name__}.{ext}", binary_images[0])
        
        for i in range(0, len(diff_images)):
            methods_diffs[i].append(diff_images[i])
            cv_methods_diffs[i].append(cv_diff_images[i])
            run_times[i].append(img_run_times[i])


    # Plot statistics if not saving files
    if save is False:
        
        # Colors for plotting
        graph_visibility_factor = max(round(img_cnt/500), 1)
        colors = ["#000000",
                  "#00a174",
                  "#ff9c00",
                  "#a1ff00",
                  "#ff0004",
                  "#6fc0ff",
                  "#e374ff",
                  "#0000ff",
                  "#aa714b"]
        xs = [i for i in range(0, img_cnt)]
        xs = xs[::graph_visibility_factor]

        
        # PSNR diff graph
        plt.rcParams["figure.figsize"] = [15, 10]
        plt.figure("PSNR diff")
        for i in range(0, len(thresh_methods.methods)):

            plt.plot(
                xs,
                gaussian_filter1d(methods_diffs[i][::graph_visibility_factor], sigma=2),
                label=thresh_methods.methods[i].__name__,
                color=colors[i]
                )
            plt.axhline(
                y = np.average(methods_diffs[i]),
                color = colors[i],
                linestyle = '--',
                label=f"{thresh_methods.methods[i].__name__} (AVG)"
                ) 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=math.ceil(len(thresh_methods.methods)/2))
        plt.grid()
        plt.tight_layout()


        # OpenCV diff graph
        plt.figure("OpenCV diff")
        plt.rcParams["figure.figsize"] = [15, 10]
        for i in range(0, len(thresh_methods.methods)):

            plt.plot(
                xs,
                gaussian_filter1d(cv_methods_diffs[i][::graph_visibility_factor], sigma=2),
                label=thresh_methods.methods[i].__name__,
                color=colors[i]
                )
            plt.axhline(
                y = np.average(cv_methods_diffs[i]),
                color = colors[i],
                linestyle = '--',
                label=f"{thresh_methods.methods[i].__name__} (AVG)"
                ) 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=math.ceil(len(thresh_methods.methods)/2))
        plt.grid()
        plt.tight_layout()

        # Runtime graph
        plt.figure("Runtime in ms")
        plt.rcParams["figure.figsize"] = [15, 10]
        for i in range(0, len(thresh_methods.methods)):

            plt.plot(
                xs,
                gaussian_filter1d(run_times[i][::graph_visibility_factor], sigma=2),
                label=thresh_methods.methods[i].__name__,
                color=colors[i]
                )
            plt.axhline(
                y = np.average(run_times[i]),
                color = colors[i],
                linestyle = '--',
                label=f"{thresh_methods.methods[i].__name__} (AVG)"
                ) 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=math.ceil(len(thresh_methods.methods)/2))
        plt.grid()
        plt.yscale("log")
        plt.tight_layout()

        plt.show()

    tear_down()
