import streamlit as st
import re
import difflib

from ocr_processor import OCRProcessor
from essay_grader import EssayGrader

@st.cache_resource
def load_grader():
    json_path = "hufs_2023_1.json" 
    return EssayGrader(json_path=json_path)

@st.cache_resource
def load_ocr_processor():
    return OCRProcessor()

def display_correction_in_streamlit(student_answer, model_answer, correction_result):
    st.subheader("🤖 AI 멘토 첨삭 결과", divider='rainbow')
    col1, col2 = st.columns(2)
    with col1:
        st.info("👨‍🎓 학생 답안 (OCR 변환)")
        st.text_area("학생 답안 내용", value=student_answer, height=300, disabled=True, key="student_answer_area")
    with col2:
        st.warning("✅ 모범 답안")
        st.text_area("모범 답안 내용", value=model_answer, height=300, disabled=True, key="model_answer_area")
    st.markdown("---")
    st.info("✨ 논리왕 김멘토's 코멘트")
    suggestions_section = re.search(r'(\*\*\[이렇게 바꿔보세요.*?💡\*\*.*)', correction_result, re.DOTALL)
    main_correction = re.split(r'\*\*\[이렇게 바꿔보세요', correction_result)[0]
    st.markdown(main_correction)
    if suggestions_section:
        st.markdown(suggestions_section.group(1))
        suggestions = re.findall(r'학생 원문: (.*?)\s*->\s*수정 제안: (.*?)(?=\n학생 원문:|\Z)', correction_result, re.DOTALL)
        for i, (original, suggestion) in enumerate(suggestions):
            original, suggestion = original.strip(), suggestion.strip()
            with st.expander(f"수정 제안 #{i+1}", expanded=True):
                st.text_input("학생 원문", value=original, disabled=True, key=f"orig_{i}")
                diff = difflib.ndiff(original.split(), suggestion.split())
                highlighted_suggestion = ""
                for word in diff:
                    if word.startswith('+ '):
                        highlighted_suggestion += f"<span style='background-color: #d4edda; padding: 2px; border-radius: 3px;'>{word[2:]}</span> "
                    elif word.startswith('- '):
                        pass
                    else:
                        highlighted_suggestion += f"{word[2:]} "
                st.markdown(f"**수정 제안:** {highlighted_suggestion}", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI 논술 첨삭 멘토", layout="wide")

    with st.sidebar:
        st.header("📝 AI 논술 첨삭 멘토")
        st.markdown("손글씨 답안 사진을 올리면 AI가 채점하고 첨삭해줘요!")
        uploaded_file = st.file_uploader("여기에 손글씨 답안 사진을 올려주세요.", type=['jpg', 'jpeg', 'png'])
        question_info = st.text_input("문제 정보를 입력하세요.", value="2023년 한국외국어대학교 인문논술 문제 2번")
        submit_button = st.button("첨삭 시작하기", type="primary")

    st.title("✍️ AI 논술 첨삭 멘토봇")
    st.markdown("왼쪽 사이드바에 답안 사진을 업로드하고 '첨삭 시작하기' 버튼을 눌러주세요.")

    if submit_button and uploaded_file:
        with st.spinner("AI 멘토가 학생 답안을 열심히 읽고 있어요... 잠시만 기다려주세요! 🧐"):
            try:
                grader = load_grader()
                ocr_processor = load_ocr_processor()

                image_bytes = uploaded_file.getvalue()
                student_answer_text = ocr_processor.process_image(image_bytes)
                
                if "문제" in student_answer_text: # OCR 실패 메시지를 더 유연하게 감지
                    st.error(student_answer_text)
                    return

                model_answer_text = grader.get_model_answer(question_info)
                correction_result = grader.grade_essay(question_info, student_answer_text)

                display_correction_in_streamlit(student_answer_text, model_answer_text, correction_result)

            except Exception as e:
                st.error(f"첨삭 과정에서 예상치 못한 오류가 발생했습니다: {e}")
    
    elif submit_button and not uploaded_file:
        st.warning("파일을 먼저 업로드해주세요!")

if __name__ == "__main__":
    main()