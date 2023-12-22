
# Calculate the NC and PSNR values
nc_ow_wi = calculate_nc(original_image, watermarked_image)
psnr_ow_wi = calculate_psnr(original_image, watermarked_image)
nc_wm_ew = calculate_nc(watermark_image, extracted_watermark)
