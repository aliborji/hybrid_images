import cv2
import numpy as np

import os
import argparse


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def convolution(img, kernel):
    """ This function executes the convolution between `img` and `kernel`.
    """
    print("[{}]\tRunning convolution...\n".format(img))
    # Load the image.
    image = cv2.imread(img)
    # Flip template before convolution.
    kernel = cv2.flip(kernel, -1)
    # Get size of image and kernel. 3rd value of shape is colour channel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    (pad_h, pad_w) = (kernel_h // 2, kernel_w // 2)
    # Create image to write to.
    output = np.zeros(image.shape)
    # Slide kernel across every pixel.
    for y in range(pad_h, image_h - pad_h):
        for x in range(pad_w, image_w - pad_w):
            # If coloured, loop for colours.
            for colour in range(image.shape[2]):
                # Get center pixel.
                center = image[
                    y - pad_h : y + pad_h + 1, x - pad_w : x + pad_w + 1, colour
                ]
                # Perform convolution and map value to [0, 255].
                # Write back value to output image.
                output[y, x, colour] = (center * kernel).sum() / 255

    # Return the result of the convolution.
    return output


def fourier(img, kernel):
    """ Compute convolution between `img` and `kernel` using numpy's FFT.
    """
    # Load the image.
    image = cv2.imread(img)
    # Get size of image and kernel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    # Apply padding to the kernel.
    padded_kernel = np.zeros(image.shape[:2])
    start_h = (image_h - kernel_h) // 2
    start_w = (image_w - kernel_w) // 2
    padded_kernel[start_h : start_h + kernel_h, start_w : start_w + kernel_w] = kernel
    # Create image to write to.
    output = np.zeros(image.shape)
    # Run FFT on all 3 channels.
    for colour in range(3):
        Fi = np.fft.fft2(image[:, :, colour])
        Fk = np.fft.fft2(padded_kernel)
        # Inverse fourier.
        output[:, :, colour] = np.fft.fftshift(np.fft.ifft2(Fi * Fk)) / 255

    # Return the result of convolution.
    return output


def gaussian_blur(image, sigma, fourier_flag):
    """ Builds a Gaussian kernel used to perform the LPF on an image.
    """
    print("[{}]\tCalculating Gaussian kernel...".format(image))

    # Calculate size of filter.
    size = 8 * sigma + 1
    if not size % 2:
        size = size + 1

    center = size // 2
    kernel = np.zeros((size, size))

    # Generate Gaussian blur.
    for y in range(size):
        for x in range(size):
            diff = (y - center) ** 2 + (x - center) ** 2
            kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    if fourier_flag:
        return fourier(image, kernel)
    else:
        return convolution(image, kernel)


def low_pass(image, cutoff, fourier):
    """ Generate low pass filter of image.
    """
    print("[{}]\tGenerating low pass image...".format(image))
    return gaussian_blur(image, cutoff, fourier)


def high_pass(image, cutoff, fourier):
    """ Generate high pass filter of image. This is simply the image minus its
    low passed result.
    """
    print("[{}]\tGenerating high pass image...".format(image))
    return (cv2.imread(image) / 255) - low_pass(image, cutoff, fourier)


def hybrid_image(image, cutoff, fourier):
    """ Create a hybrid image by summing together the low and high frequency
    images.
    """
    # Perform low pass filter and export.
    low = low_pass(image[0], cutoff[0], fourier)
    cv2.imwrite("low.jpg", low * 255)
    # Perform high pass filter and export.
    high = high_pass(image[1], cutoff[1], fourier)
    cv2.imwrite("high.jpg", (high + 0.5) * 255)

    print("Creating hybrid image...")
    return low + high


def output_vis(image):
    """ Display hybrid image comparison for report. Visualisation shows 5 images
    reducing in size to simulate viewing the image from a distance.
    """
    print("Creating visualisation...")

    num = 5  # Number of images to display.
    gap = 2  # Gap between images (px).

    # Create list of images.
    image_list = [image]
    max_height = image.shape[0]
    max_width = image.shape[1]

    # Add images to list and increase max width.
    for i in range(1, num):
        tmp = cv2.resize(image, (0, 0), fx=0.5 ** i, fy=0.5 ** i)
        max_width += tmp.shape[1] + gap
        image_list.append(tmp)

    # Create space for image stack.
    stack = np.ones((max_height, max_width, 3)) * 255

    # Add images to stack.
    current_x = 0
    for img in image_list:
        stack[
            max_height - img.shape[0] :, current_x : img.shape[1] + current_x, :
        ] = img
        current_x += img.shape[1] + gap

    return stack



def mass_hybrid(config):
    """ Create hybrid image from two source images. xxxx
    """
    # print("hi")
    indir = '/Users/aliborji/Desktop/DB/'
    outdir = '/Users/aliborji/Desktop/DB/Output/'
    folders = ['Banana', 'CustardApple', 'Fig', 'GrannySmith', 'Jackfruit', 'Lemon', 'Pineapple', 'Pomegranate', 'Strawberry', 'Orange']
    
    count = 0
    for i1, f1 in enumerate(folders):
        for i2, f2 in enumerate(folders):
            if f1 == f2:
                continue

            f1_files = os.listdir(os.path.join(indir, f1))    
            f2_files = os.listdir(os.path.join(indir, f2))

            for file1 in f1_files:
                if not file1.endswith('.jpeg'): continue
                for file2 in f2_files:
                    if not file2.endswith('.jpeg'): continue    

                    for cut in range(1,21,3):

                        hybrid = hybrid_image([os.path.join(indir, f1, file1), os.path.join(indir, f2, file2)], [cut, cut], config.fourier)
                        count += 1

                        # fname = '_'.join([file1[:-5], file2[:-5], str(cut), str(count), '.jpg'])    
                        fname = '_'.join([str(i1), str(i2), str(cut), str(count), '.jpg'])                            
                        cv2.imwrite(os.path.join(outdir, fname), hybrid * 255)


def hybrid(config):
    """ Create hybrid image from two source images. xxxx
    """

    for cut in range(1,21,3):

        hybrid = hybrid_image([config.i1, config.i0], [cut, cut], config.fourier)

        # import pdb; pdb.set_trace()
        if config.visual:
            cv2.imwrite('_'+str(cut)+'_'+ config.output, output_vis(hybrid) * 255)
        else:
            cv2.imwrite('_'+str(cut)+'_'+config.output, hybrid * 255)


    for cut in range(1,21,3):

        hybrid = hybrid_image([config.i0, config.i1], [cut, cut], config.fourier)

        # import pdb; pdb.set_trace()
        if config.visual:
            cv2.imwrite(str(cut)+'_'+ config.output, output_vis(hybrid) * 255)
        else:
            cv2.imwrite(str(cut)+'_'+config.output, hybrid * 255)




if __name__ == "__main__":
    # print("hi")
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('-i0', type=str)
    parser.add_argument('-i1', type=str)    
    parser.add_argument("-o", "--output", default="output.jpg", help="Output file.")
    parser.add_argument(
        "-c",
        "--cutoff",
        default=[10,10],
        type=int,
        help="High/low cutoff frequencies.",
    )
    parser.add_argument(
        "-v", "--visual", action='store_false', help="Generate visualisation."
    )
    parser.add_argument(
        "-f", "--fourier", action='store_false', help="Use fourier convolution."
    )

    config = parser.parse_args()

    # mass_hybrid(config)
    hybrid(config)
