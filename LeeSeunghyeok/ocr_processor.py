# ocr_processor.py (사용자 성공 버전 기반 최종 완성본)

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io

class OCRProcessor:
    def __init__(self):
        """
        사용자가 성공한 가장 단순하고 안정적인 방식으로 PaddleOCR 모델을 초기화합니다.
        """
        print("PaddleOCR 모델을 로딩합니다... (사용자 성공 버전 기반)")
        # 사용자가 성공한 가장 단순한 초기화 방식을 그대로 적용합니다.
        # 모든 부가 옵션을 제거한 것이 안정성의 핵심이었습니다.
        self.ocr = PaddleOCR(lang="korean")
        print("✅ PaddleOCR 모델 로딩 완료!")

    def process_image(self, image_source):
        """
        이미지 바이트를 입력받아, 안정적인 RGB 포맷으로 변환 후 텍스트를 추출합니다.
        """
        try:
            # 1. 흑백/알파채널 문제를 방지하기 위해 이미지를 RGB로 강제 변환합니다.
            #    (이건 나중에 다른 이미지에서 생길 문제를 예방하는 좋은 습관입니다.)
            image_file = io.BytesIO(image_source)
            image = Image.open(image_file).convert('RGB')
            np_array = np.array(image)

            # 2. OCR 실행
            #    사용자의 성공 코드처럼, ocr() 함수는 이미지 경로뿐만 아니라
            #    numpy 배열도 처리할 수 있습니다.
            result = self.ocr.ocr(np_array)

            # 3. 결과 처리
            #    결과가 없거나 비어있는 경우를 방어합니다.
            if not result or not result[0]:
                return "이미지에서 텍스트를 추출하지 못했습니다."

            # line_info는 [좌표, ('텍스트', 정확도)] 형태일 것입니다.
            # text_lines = [line_info[1][0] for line_info in result[0]]
            ocr_output_dict = result[0]
            text_lines = ocr_output_dict.get('rec_texts', []) # .get()을 사용하면 'rec_texts' 키가 없어도 오류 없이 안전하게 빈 리스트를 반환합니다.
            
            # ocr_result = result[0]
            # print('\n'.join(ocr_result['rec_texts']))

            # 성공적으로 추출된 텍스트들을 하나의 문자열로 합쳐 반환합니다.
            return "\n".join(text_lines)
            # return "\n".join(ocr_result)

        except Exception as e:
            print(f"[OCR-ERROR] 처리 중 예상치 못한 오류 발생: {e}")
            return f"OCR 처리 중 문제가 발생했습니다. 관리자에게 문의하세요."