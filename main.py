import DWT_DCT
import cv2 # for reading and writing images
from Evaluation_parameters import calculate_nc,calculate_psnr

original_image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
watermark_image = cv2.imread('watermark.jpg', cv2.IMREAD_GRAYSCALE)

dwt_dct = DWT_DCT.DWT_DCT(original_image, watermark_image)

watermarked_image = dwt_dct.embed()

# Save the watermarked image
cv2.imwrite('./output/watermarked_image.png', watermarked_image)

#extraction

extracted_watermark = dwt_dct.extract(watermarked_image)

# Save the extracted watermark
cv2.imwrite('./output/extracted_watermark.png', extracted_watermark)

# Calculate the NC for original and watermarked images
print("NC between original image and watermarked image: ", calculate_nc(original_image, watermarked_image))

# Calculate the PSNR for original and watermarked images
print("PSNR between original image and watermarked image: ", calculate_psnr(original_image, watermarked_image))

# Calculate the NC for original and watermarked images
print("NC between original watermark and extracted watermark: ", calculate_nc(watermark_image, extracted_watermark))





