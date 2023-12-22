import cv2 # for reading and writing images
import numpy as np # for mathematical operations
import pywt # for discrete wavelet transform
from Evaluation_parameters import calculate_nc, calculate_psnr

class DWT_DCT: 
    def __init__(self, original_image, watermark_image, alpha = 0.01):
        self.original_image = original_image
        self.watermark_image = watermark_image
        self.alpha = alpha # This should be the same alpha used during the embedding process
        
    
    def decompose1(self, image):
        #Apply DWT on the orignal image
        coeffs = pywt.dwt2(image, 'haar')
        LL, (LH, HL, HH) = coeffs

        return LL, (LH, HL, HH)

    def decomposeLL2(self, LL):
         # Apply DWT on the decomposed image subband LL to get LL2
        coeffs2 = pywt.dwt2(LL, 'haar')
        LL2, (LH2, HL2, HH2) = coeffs2

        return LL2, (LH2, HL2, HH2)
    
    def dct_apply(self, selected_coeffs):
        selected_coeffs = np.float32(selected_coeffs) # we convert the image to float32 because dct function works on floating points 
        selected_coeffs_dct = cv2.dct(selected_coeffs)
        return selected_coeffs_dct

    def embed(self):
        """ 
        1- Apply DWT on The original Image
        2- Apply DWT on the decomposed image subband LL to get LL2
        3- Apply DCT on the decomposed image subband LL2
        4- Apply IDCT on the decomposed image subband LL2
        5- Apply IDWT on The LL2 to get LL and then apply IDWT on LL to get the watermarked image
        6- Calculate the PSNR
        7- Calculate the NC 
        """
        coeffs = self.decompose1(self.original_image)
        LL, (LH, HL, HH) = coeffs
        coeffs2 = self.decomposeLL2(LL)
        LL2, (LH2, HL2, HH2) = coeffs2
        LL2_dct = self.dct_apply(LL2)

        # embed the watermark in the DCT coefficients
        # Resize the watermark to match the LL2 sub-band dimensions
        watermark_resized = cv2.resize(self.watermark_image, (LL2_dct.shape[1], LL2_dct.shape[0]))

        # Spread spectrum watermarking
        alpha = 0.01  # This is the embedding strength factor
        LL2_dct_watermarked = LL2_dct + alpha * cv2.dct(np.float32(watermark_resized))

        # Apply inverse DCT and DWT
        watermarked_LL2 = cv2.idct(LL2_dct_watermarked)
        watermarked_subband_LL = pywt.idwt2((watermarked_LL2, (LH2, HL2, HH2)), 'haar')
        watermarked_image = pywt.idwt2((watermarked_subband_LL, (LH, HL, HH)), 'haar')

        return watermarked_image



    def extract(self, watermarked_image):
        originalcoeffs = self.decompose1(self.original_image)
        originalLL, (originalLH, originalHL, originalHH) = originalcoeffs
        originalcoeffs2 = self.decomposeLL2(originalLL)
        originalLL2, (originalLH2, originalHL2, originalHH2) = originalcoeffs2

        originalLL2_dct = self.dct_apply(originalLL2)

        watermarkedcoeffs = self.decompose1(watermarked_image)
        watermarkedLL, (watermarkedLH, watermarkedHL, watermarkedHH) = watermarkedcoeffs

        watermarkedcoeffs2 = self.decomposeLL2(watermarkedLL)
        watermarkedLL2, (watermarkedLH2, watermarkedHL2, watermarkedHH2) = watermarkedcoeffs2

        watermarkedLL2_dct = self.dct_apply(watermarkedLL2)

        #The difference between the DCT coefficients of the original image and the watermarked image
        dct_diff = watermarkedLL2_dct - originalLL2_dct

        alpha = 0.01 # This should be the same alpha used during the embedding process
        watermark = dct_diff / alpha

        watermark = cv2.idct(watermark)
        watermark_normalized = cv2.normalize(watermark, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return watermark_normalized







