import cv2 # for reading and writing images
import numpy as np # for mathematical operations
import pywt # for discrete wavelet transform
from Evaluation_parameters import calculate_nc, calculate_psnr


# Load the original and watermarked images
original_image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
watermarked_image = cv2.imread('watermarked_image.png', cv2.IMREAD_GRAYSCALE)



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
