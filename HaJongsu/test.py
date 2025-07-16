from paddleocr import PaddleOCR
# from paddleocr.utils import draw_ocr
from PIL import Image
import matplotlib.pyplot as plt
from pprint import pprint

# 1. OCR 모델 초기화 (한국어 + 영어 지원)
ocr = PaddleOCR(lang='korean')  # lang='korean' 또는 'korean+english'

# 2. 이미지 파일 경로
image_path = './hand_writing.jpeg'

# 3. 이미지에서 텍스트 인식
results = ocr.ocr(image_path)

ocr_result = results[0]
pprint(ocr_result['rec_texts'])


# # 4. 추출된 텍스트 출력
# for line in results:
#     for (box, text, confidence) in line:
#         print(f"인식된 텍스트: {text}, 신뢰도: {confidence:.2f}")

# # 5. 시각화 (선택 사항)
# image = Image.open(image_path).convert('RGB')
# boxes = [element[0] for line in results for element in line]
# txts = [element[1][0] for line in results for element in line]
# scores = [element[1][1] for line in results for element in line]

# image_with_boxes = draw_ocr(image, boxes, txts, scores)
# plt.imshow(image_with_boxes)
# plt.axis('off')
# plt.title("PaddleOCR 결과 시각화")
# plt.show()
