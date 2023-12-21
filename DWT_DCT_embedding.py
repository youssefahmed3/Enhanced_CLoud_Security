import cv2 # for reading and writing images
import numpy as np # for mathematical operations
import pywt # for discrete wavelet transform

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

def calculate_nc(image, template):
    # Ensure the images are the same size
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


# Load the original and watermarked images
original_image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
watermarked_image = cv2.imread('watermarked_image.png', cv2.IMREAD_GRAYSCALE)
""" 
1- Apply DWT on The Image
2- Apply DWT on the decomposed image subband LL to get LL2
3- Apply DCT on the decomposed image subband LL2
4- Apply IDCT on the decomposed image subband LL2
5- Apply IDWT on The LL2 to get LL and then apply IDWT on LL to get the watermarked image
6- Calculate the PSNR
7- Calculate the NC 
"""

#load the original image
original_image = cv2.imread('lena.png')
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

#load the watermark image
watermark_image = cv2.imread('watermark.jpg')
watermark_image_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)

#Apply DWT on the orignal image
coeffs = pywt.dwt2(original_image_gray, 'haar')
LL, (LH, HL, HH) = coeffs

# Apply DWT on the decomposed image subband LL to get LL2
coeffs2 = pywt.dwt2(LL, 'haar')
LL2, (LH2, HL2, HH2) = coeffs2



# Apply DCT on the decomposed image subband LL2
LL2 = np.float32(LL2) # we convert the image to float32 because dct function works on floating points 
LL2_dct = cv2.dct(LL2)

# embed the watermark in the DCT coefficients
# Resize the watermark to match the LL2 sub-band dimensions
watermark_resized = cv2.resize(watermark_image_gray, (LL2_dct.shape[1], LL2_dct.shape[0]))

# Spread spectrum watermarking
alpha = 0.01  # This is the embedding strength factor
LL2_dct_watermarked = LL2_dct + alpha * cv2.dct(np.float32(watermark_resized))

# Apply inverse DCT and DWT
watermarked_LL2 = cv2.idct(LL2_dct_watermarked)
watermarked_subband_LL = pywt.idwt2((watermarked_LL2, (LH2, HL2, HH2)), 'haar')
watermarked_image = pywt.idwt2((watermarked_subband_LL, (LH, HL, HH)), 'haar')

# Save the watermarked image
cv2.imwrite('watermarked_image.png', watermarked_image)
print("Watermarked image saved successfully")

# Calculate the PSNR
print("PSNR: ", calculate_psnr(original_image_gray, watermarked_image))

# Calculate the NC

print("NC: ", calculate_nc(original_image_gray, watermarked_image))
