# ocr_processor.py

# 필요한 라이브러리 임포트
from paddleocr import PaddleOCR
import numpy as np

# OCR 처리를 담당하는 클래스 정의
class OCRProcessor:
    """
    PaddleOCR을 사용하여 이미지에서 텍스트를 추출하는 클래스.
    """
    # 클래스가 생성될 때 초기 설정을 하는 함수
    def __init__(self):
        """
        PaddleOCR 모델 초기화.
        - lang='korean': 한국어와 영어를 함께 인식 ('en'도 포함됨)
        - use_gpu=True: GPU 사용 설정 (GPU가 없으면 자동으로 CPU 사용)
        """
        print("PaddleOCR 모델을 로딩합니다...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=True)
        print("✅ PaddleOCR 모델 로딩 완료!")

    # 이미지를 받아 텍스트를 추출하는 함수
    def process_image(self, image_source):
        """
        이미지 바이트(bytes)를 입력받아 텍스트를 추출.

        Args:
            image_source (bytes): 사용자가 업로드한 이미지 파일의 바이트 데이터.

        Returns:
            str: 이미지에서 추출된 텍스트 전체. 오류 발생 시 오류 메시지 반환.
        """
        try:
            # 이미지 바이트를 numpy 배열로 변환
            # 웹에서 바로 받은 이미지 데이터 처리에 적합
            np_array = np.frombuffer(image_source, np.uint8)
            
            # PaddleOCR로 텍스트 인식 실행
            # self.ocr.ocr()은 이미지 데이터와 텍스트 분류(cls) 여부를 인자로 받음
            result = self.ocr.ocr(np_array, cls=True)
            
            # 결과가 비어있는 경우 처리
            if not result or not result[0]:
                return "이미지에서 텍스트를 추출하지 못했습니다."

            # 추출된 텍스트 조각들을 리스트로 만듦
            # result 구조: [[[[좌표], [좌표], ...], ('텍스트', 정확도)], ...]
            # 여기서 '텍스트' 부분만 추출
            text_lines = [line[1][0] for line in result[0]]
            
            # 텍스트 라인들을 하나의 문자열로 결합
            return "\n".join(text_lines)

        except Exception as e:
            # 오류 발생 시 사용자에게 알려줄 메시지
            return f"OCR 처리 중 오류가 발생했습니다: {e}"

# 이 파일이 메인으로 실행될 때만 아래 테스트 코드를 실행
if __name__ == '__main__':
    # 테스트용 이미지 파일 경로 (실제 파일 경로로 수정 필요)
    # 예시: test_image.png
    image_path = "YOUR_TEST_IMAGE_PATH.jpg" # 여기에 테스트할 이미지 파일 경로를 넣어봐!

    try:
        # OCRProcessor 객체 생성
        processor = OCRProcessor()
        
        # 이미지 파일을 바이너리 읽기 모드로 열기
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # 이미지 처리 및 텍스트 추출
        extracted_text = processor.process_image(image_bytes)
        
        print("\n--- OCR 추출 결과 ---")
        print(extracted_text)

    except FileNotFoundError:
        print(f"'{image_path}' 파일을 찾을 수 없습니다. 테스트를 위해 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")