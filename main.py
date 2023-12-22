import DWT_DCT
import cv2 # for reading and writing images
from Evaluation_parameters import calculate_nc,calculate_psnr
import matplotlib.pyplot as plt

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



# Calculate the NC and PSNR values
nc_ow_wi = calculate_nc(original_image, watermarked_image)
psnr_ow_wi = calculate_psnr(original_image, watermarked_image)
nc_wm_ew = calculate_nc(watermark_image, extracted_watermark)
# Plot the original and watermarked images
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Base Image')

plt.subplot(2, 2, 2)
plt.imshow(watermark_image, cmap='gray')
plt.title('Watermark')

plt.subplot(2, 2, 3)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')


plt.subplot(2, 2, 4)
plt.imshow(extracted_watermark, cmap='gray')
plt.title('extracted watermark')

# Add the labels to the bottom of the figure
plt.figtext(0.5, 0.01, f"NC between original image and watermarked image: {nc_ow_wi}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.figtext(0.5, 0.06, f"PSNR between original image and watermarked image: {psnr_ow_wi}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.figtext(0.5, 0.11, f"NC between original watermark and extracted watermark: {nc_wm_ew}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.show()


