# main.py

# --- 필요한 라이브러리 임포트 ---

# 웹 UI를 만들기 위한 Streamlit 라이브러리
import streamlit as st

# 텍스트 비교 및 시각화를 위한 라이브러리
import re
import difflib

# 우리가 만든 모듈 임포트
from ocr_processor import OCRProcessor
from essay_grader import EssayGrader

# --- Streamlit 캐싱 기능 활용 ---
# @st.cache_resource 데코레이터는 복잡하고 시간이 오래 걸리는 객체 생성을
# 한 번만 실행하고 그 결과를 계속 재사용하게 해주는 마법 같은 기능이야.
# 사용자가 앱과 상호작용할 때마다 모델을 새로 로딩하는 것을 막아줘서 속도가 매우 빨라져.

@st.cache_resource
def load_grader():
    """ EssayGrader 객체를 로드하고 캐싱합니다. """
    # 여기에 논술 자료 JSON 파일 경로를 지정해줘!
    # 여러 개의 파일을 다루고 싶다면, 이 부분을 나중에 확장할 수 있어.
    json_path = "hufs_2023_1.json" 
    return EssayGrader(json_path=json_path)

@st.cache_resource
def load_ocr_processor():
    """ OCRProcessor 객체를 로드하고 캐싱합니다. """
    return OCRProcessor()

# --- 첨삭 결과 시각화 함수 ---
# Colab의 display/HTML 코드를 Streamlit에 맞게 수정한 버전
def display_correction_in_streamlit(student_answer, model_answer, correction_result):
    """
    학생 답안, 모범 답안, AI 첨삭 결과를 Streamlit 화면에 보기 좋게 출력.
    """
    st.subheader("🤖 AI 멘토 첨삭 결과", divider='rainbow')

    # 1. 학생 답안과 모범 답안을 나란히 배치 (두 개의 컬럼 사용)
    col1, col2 = st.columns(2)
    with col1:
        st.info("👨‍🎓 학생 답안 (OCR 변환)")
        st.text_area("학생 답안 내용", value=student_answer, height=300, disabled=True)
    with col2:
        st.warning("✅ 모범 답안")
        st.text_area("모범 답안 내용", value=model_answer, height=300, disabled=True)

    st.markdown("---")

    # 2. AI 멘토의 종합 첨삭 결과 출력
    st.info("✨ 논리왕 김멘토's 코멘트")

    # '이렇게 바꿔보세요' 부분을 분리하여 특별하게 처리
    suggestions_section = re.search(r'(\*\*\[이렇게 바꿔보세요.*?💡\*\*.*)', correction_result, re.DOTALL)
    main_correction = re.split(r'\*\*\[이렇게 바꿔보세요', correction_result)[0]
    
    # 기본 첨삭 내용 출력
    st.markdown(main_correction)

    # 3. 문장 수정 제안 시각화 (diff 기능)
    if suggestions_section:
        st.markdown(suggestions_section.group(1)) # "이렇게 바꿔보세요" 제목 출력
        
        # "학생 원문: ... -> 수정 제안: ..." 형식의 모든 제안을 찾음
        suggestions = re.findall(r'학생 원문: (.*?)\s*->\s*수정 제안: (.*?)(?=\n학생 원문:|\Z)', correction_result, re.DOTALL)
        
        for i, (original, suggestion) in enumerate(suggestions):
            original = original.strip()
            suggestion = suggestion.strip()
            
            with st.expander(f"수정 제안 #{i+1}", expanded=True):
                st.text_input("학생 원문", value=original, disabled=True)
                
                # difflib를 사용해 변경된 부분을 하이라이트
                diff = difflib.ndiff(original.split(), suggestion.split())
                highlighted_suggestion = ""
                for word in diff:
                    if word.startswith('+ '):
                        highlighted_suggestion += f"<span style='background-color: #d4edda; padding: 2px; border-radius: 3px;'>{word[2:]}</span> "
                    elif word.startswith('- '):
                        pass # 삭제된 단어는 표시하지 않음
                    else:
                        highlighted_suggestion += f"{word[2:]} "
                
                st.markdown(f"**수정 제안:** {highlighted_suggestion}", unsafe_allow_html=True)


# --- 메인 앱 실행 함수 ---
def main():
    # Streamlit 페이지 기본 설정
    st.set_page_config(page_title="AI 논술 첨삭 멘토", layout="wide")

    # --- 사이드바 UI ---
    with st.sidebar:
        st.header("📝 AI 논술 첨삭 멘토")
        st.markdown("손글씨 답안 사진을 올리면 AI가 채점하고 첨삭해줘요!")
        
        # 파일 업로드 기능. 'jpg', 'jpeg', 'png' 형식만 허용
        uploaded_file = st.file_uploader("여기에 손글씨 답안 사진을 올려주세요.", type=['jpg', 'jpeg', 'png'])
        
        # 문제 정보 입력 (나중에는 DB에서 선택하도록 확장 가능)
        question_info = st.text_input(
            "문제 정보를 입력하세요.", 
            value="2023년 한국외국어대학교 인문논술 문제 2번"
        )
        
        # 첨삭 시작 버튼
        submit_button = st.button("첨삭 시작하기", type="primary")

    # --- 메인 화면 UI ---
    st.title("✍️ AI 논술 첨삭 멘토봇")
    st.markdown("왼쪽 사이드바에 답안 사진을 업로드하고 '첨삭 시작하기' 버튼을 눌러주세요.")

    # 사용자가 버튼을 눌렀을 때 실행될 로직
    if submit_button and uploaded_file:
        with st.spinner("AI 멘토가 학생 답안을 열심히 읽고 있어요... 잠시만 기다려주세요! 🧐"):
            try:
                # 1. 필요한 모델/클래스 로드 (캐싱 덕분에 빠름)
                grader = load_grader()
                ocr_processor = load_ocr_processor()

                # 2. 이미지 파일 OCR 처리
                st.write("1단계: 이미지에서 글자를 추출하는 중... (OCR)")
                image_bytes = uploaded_file.getvalue()
                student_answer_text = ocr_processor.process_image(image_bytes)
                
                if "오류" in student_answer_text or not student_answer_text.strip():
                    st.error(f"OCR 처리 중 문제가 발생했습니다: {student_answer_text}")
                    return

                # 3. 모범 답안 검색 및 AI 첨삭 실행
                st.write("2단계: 모범 답안과 비교하며 첨삭하는 중... (RAG)")
                model_answer_text = grader.get_model_answer(question_info)
                correction_result = grader.grade_essay(question_info, student_answer_text)

                # 4. 결과 출력
                st.write("3단계: 첨삭 결과를 생성하는 중... ✨")
                display_correction_in_streamlit(student_answer_text, model_answer_text, correction_result)

            except Exception as e:
                st.error(f"첨삭 과정에서 예상치 못한 오류가 발생했습니다: {e}")
    
    elif submit_button and not uploaded_file:
        st.warning("파일을 먼저 업로드해주세요!")

# 이 파일이 직접 실행될 때 main() 함수를 호출
if __name__ == "__main__":
    main()