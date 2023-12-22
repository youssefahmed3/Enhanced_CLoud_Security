import cv2 # for reading and writing images
import numpy as np # for mathematical operations
import pywt # for discrete wavelet transform
from Evaluation_parameters import calculate_nc

#load the original watermark 
original_watermark = cv2.imread('watermark.jpg', cv2.IMREAD_GRAYSCALE)

#load the original image
original_image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

#apply DWT on the original image
original_image_coeffs = pywt.dwt2(original_image, 'haar')
original_LL, (original_LH, original_HL, original_HH) = original_image_coeffs

#apply DWT on the LL subband of the original image
original_LL_coeffs = pywt.dwt2(original_LL, 'haar')
original_LL2, (original_LH2, original_HL2, original_HH2) = original_LL_coeffs

# Apply DCT on the decomposed original image subband LL2
original_LL2 = np.float32(original_LL2) # we convert the image to float32 because dct function works on floating points 
original_LL2_dct = cv2.dct(original_LL2)




#load the watermarked image
watermarked_image = cv2.imread('watermarked_image.png', cv2.IMREAD_GRAYSCALE)

#apply DWT on the watermarked image
watermarked_image_coeffs = pywt.dwt2(watermarked_image, 'haar')
watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = watermarked_image_coeffs

#apply DWT on the LL subband of the watermarked image
watermarked_LL_coeffs = pywt.dwt2(watermarked_LL, 'haar')
watermarked_LL2, (watermarked_LH2, watermarked_HL2, watermarked_HH2) = watermarked_LL_coeffs

# Apply DCT on the decomposed watermarked image subband LL2
watermarked_LL2 = np.float32(watermarked_LL2) # we convert the image to float32 because dct function works on floating points 
watermarked_LL2_dct = cv2.dct(watermarked_LL2)

#The difference between the DCT coefficients of the original image and the watermarked image
dct_diff = watermarked_LL2_dct - original_LL2_dct

alpha = 0.01 # This should be the same alpha used during the embedding process
watermark = dct_diff / alpha

watermark = cv2.idct(watermark)
watermark_normalized = cv2.normalize(watermark, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imwrite('watermark_extraction.png', watermark)
""" 
# Subtract the original image from the watermarked image
difference_image = cv2.absdiff(original_image, watermarked_image)

# Normalize the difference image for viewing
difference_image_normalized = cv2.normalize(difference_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Save and view the difference image
cv2.imwrite('difference_image.png', difference_image_normalized) """

print("NC: ", calculate_nc(original_watermark, watermark))