from PIL import Image
import io

from paddleocr import PaddleOCR
import numpy as np

class OCRProcessor:
    def __init__(self):
        print("PaddleOCR 모델을 로딩합니다... (안정화 모드)")
        self.ocr = PaddleOCR(lang='korean')
        print("✅ PaddleOCR 모델 로딩 완료!")

    def process_image(self, image_source):
        """
        이미지 바이트(bytes)를 입력받아 텍스트를 추출합니다.
        - 어떤 이미지 형식이든 3채널 RGB로 변환하여 안정성을 확보합니다.
        """
        try:
            # 이미지를 Pillow로 열고, 'RGB' 모드로 강제 변환합니다.
            # 1. 이미지 바이트를 메모리 상의 파일처럼 다룹니다.
            image_file = io.BytesIO(image_source)
            # 2. Pillow를 사용해 이미지를 엽니다.
            image = Image.open(image_file)
            # 3. 흑백이든, 투명 배경이든, 무조건 3채널 RGB 컬러로 변환합니다.
            rgb_image = image.convert('RGB')
            # 4. 변환된 이미지를 PaddleOCR이 좋아하는 numpy 배열로 바꿉니다.
            np_array = np.array(rgb_image)
            
            # 이제 안전하게 변환된 이미지를 OCR에 전달합니다.
            result = self.ocr.ocr(np_array)

            if not result or not result[0]:
                return "이미지에서 텍스트를 추출하지 못했습니다."

            text_lines = []
            for line_group in result:
                for line_info in line_group:
                    if len(line_info) > 1 and isinstance(line_info[1], (tuple, list)) and len(line_info[1]) > 0:
                        text = line_info[1][0]
                        text_lines.append(text)
            
            if text_lines:
                return "\n".join(text_lines)
            else:
                return "이미지에서 텍스트를 추출하지 못했습니다. (내용 없음)"

        except Exception as e:
            print(f"[OCR-ERROR] 처리 중 예상치 못한 오류 발생: {e}")
            return f"OCR 처리 중 문제가 발생했습니다. 관리자에게 문의하세요."