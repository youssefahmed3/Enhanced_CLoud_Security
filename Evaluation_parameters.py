import cv2 # for reading and writing images
import numpy as np # for mathematical operations

def calculate_nc(image, template):
    # Ensure the images are the same size
    if image.shape != template.shape:
        # Resize the template to match the image
        template = cv2.resize(template, (image.shape[1], image.shape[0]))

    assert image.shape == template.shape

    # Calculate the mean of the images
    image_mean = np.mean(image)
    template_mean = np.mean(template)

    # Subtract the mean from the images
    image_zero_mean = image - image_mean
    template_zero_mean = template - template_mean

    # Calculate the numerator (the sum of the product of the zero-mean images)
    numerator = np.sum(image_zero_mean * template_zero_mean)

    # Calculate the denominator (the square root of the sum of the squares of the zero-mean images)
    denominator = np.sqrt(np.sum(image_zero_mean**2) * np.sum(template_zero_mean**2))

    # Calculate the normalized correlation
    nc = numerator / denominator

    return nc


def calculate_psnr(original_image, reconstructed_image):
    # Ensure the images are the same size
    assert original_image.shape == reconstructed_image.shape

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((original_image - reconstructed_image) ** 2)

    # If the MSE is zero, the images are identical, and the PSNR is infinity
    if mse == 0:
        return float('inf')

    # Otherwise, calculate the PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr
