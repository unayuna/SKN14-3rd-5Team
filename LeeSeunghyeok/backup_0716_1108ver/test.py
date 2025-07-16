from paddleocr import PaddleOCR
 
ocr = PaddleOCR(lang="korean")
 
image_path = "test_image.png"
result = ocr.ocr(image_path)
 
ocr_result = result[0]
print('\n'.join(ocr_result['rec_texts']))