from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from config import UNIVERSITY_DATA
from essay_grader import EssayGrader
from display_ui import display_correction_with_diff
import numpy as np
import streamlit as st
import fitz  # PDF 미리보기용
import cv2, os


load_dotenv()

# 페이지 설정 (가장 첫 줄)
if "page_config_set" not in st.session_state:
    st.set_page_config(page_title="AI 논술 첨삭", layout="wide")
    st.session_state.page_config_set = True


@st.cache_resource
def load_grader():
    return EssayGrader()

grader = load_grader()

# PaddleOCR 초기화
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='korean')
ocr_model = load_ocr()

    
# def chat_with_gpt(question_id, prompt_text, history=[]):
#     # 일반 챗봇 질문 대응을 위해 EssayGrader에 단순한 채팅 기능 추가
#     return grader.graded_chat(question_id, prompt_text)
# def chat_with_gpt(prompt_text, history=[]):
#     response = grader.simple_chat(prompt_text, history)
#     st.session_state.chat_history.append({"user": prompt_text, "assistant": response})
#     return response

def render_home():
    st.markdown("""
        <style>
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding-top: 13vh;
            text-align: center;
        }

        .home-title {
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }

        .home-subtitle {
            font-size: 18px;
            color: gray;
            margin-bottom: 30px;
            text-align: center;
        }

        .button-row {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 5px;
        }

        .custom-button {
            background-color: #2d6cdf;
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            min-width: 120px;
        }

        .custom-button:hover {
            background-color: #1c4fad;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='center-container'>", unsafe_allow_html=True)
    st.markdown("<div class='home-title'>🤖 AI 첨삭 챗봇</div>", unsafe_allow_html=True)
    st.markdown("<div class='home-subtitle'>원하는 기능을 선택하세요</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📄 시험지 보기"):
            st.session_state.page = "exam"
            st.rerun()
    with col2:
        if st.button("✏️ 답안 첨삭하기"):
            st.session_state.page = "grading"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_exam():
    st.title("📝 시험지 보기")
    if st.button("🏠 홈으로", key="back_home_exam"):
        st.session_state.page = "home"
        st.rerun()

    selected_univ = st.selectbox("학교 선택", ["선택"] + list(UNIVERSITY_DATA.keys()))
    selected_year = st.selectbox("연도 선택", ["선택"] + list(UNIVERSITY_DATA.get(selected_univ, {}).keys()))
    question_keys = list(UNIVERSITY_DATA.get(selected_univ, {}).get(selected_year, {}).keys())
    selected_question = st.selectbox("문항 선택", ["선택"] + question_keys)
    if 'pix' not in st.session_state:
        st.session_state.pix = False

    if selected_univ == "선택" or selected_year == "선택" or selected_question == "선택":
        st.write('문제를 선택해주세요')
    else:
        pdf_path = UNIVERSITY_DATA[selected_univ][selected_year][selected_question]["pdf"]
        page_list = UNIVERSITY_DATA[selected_univ][selected_year][selected_question]["page"]
        st.session_state.page_list = page_list
        st.session_state['question_id'] = pdf_path.split('/')[-1].split('.')[0]
        st.session_state.selected_question = selected_question

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(page_list)
            cur_page = st.session_state.page_num

            st.markdown(f"**페이지 {cur_page + 1} / {total_pages}**")
            page = doc[page_list[cur_page]]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            st.image(pix.tobytes("png"), use_container_width=True)

            col1, _, col2 = st.columns([1, 6, 1])
            with col1:
                if st.button("⬅ 이전", key="prev_exam") and cur_page > 0:
                    st.session_state.page_num -= 1
                    st.rerun()
            with col2:
                if st.button("다음 ➡", key="next_exam") and cur_page < total_pages - 1:
                    st.session_state.page_num += 1
                    st.rerun()

        except Exception as e:
            st.error(f"PDF 로딩 실패: {e}")


def render_grading():
    st.title("✏ GPT 기반 손글씨 첨삭")

    uploaded_files = []
    with st.sidebar:
        st.markdown("### 📁 파일 업로드")
        uploaded_files = st.file_uploader(
            "다음 형식의 파일을 업로드해주세요",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if not uploaded_files:
            st.info("이미지 파일을 하나 이상 업로드해주세요.")

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]

        selected_names = st.multiselect(
            "파일을 선택해주세요 (최대 5개)",
            options=file_names,
            default=file_names[:1],
            max_selections=5
        )

        selected_files = [f for f in uploaded_files if f.name in selected_names]
        if 'extracted_text' not in st.session_state:
            st.session_state.extracted_text = False
        if selected_files:
            st.markdown("### 📸 선택한 답안지 미리보기 (가로 슬라이드)")

            index = st.session_state.slide_index
            max_index = len(selected_files) - 1

            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                if st.button("⬅️ 이전", key="prev_slide") and index > 0:
                    st.session_state.slide_index -= 1
                    st.rerun()
            with col3:
                if st.button("➡️ 다음", key="next_slide") and index < max_index:
                    st.session_state.slide_index += 1
                    st.rerun()

            current_file = selected_files[index]
            image = Image.open(current_file)
            st.image(image, caption=f"{current_file.name} ({index + 1}/{len(selected_files)})", use_container_width=True)

            with st.expander("🔍 이미지 확대 보기"):
                st.image(image, use_container_width=True)

            if st.button("🤖 GPT 첨삭 실행", key=f"gpt_feedback_{index}"):
                st.markdown("## 📄 첨삭 결과")



                # 이미지 전처리
                image = Image.open(current_file).convert('RGB')
                img_np = np.array(image)

                # OCR 수행
                try:
                    result = ocr_model.ocr(img_np)

                    if result:
                        extracted_text = '\n'.join(result[0]['rec_texts'])
                    else:
                        extracted_text = "❌ 인식된 텍스트가 없습니다."

                except Exception as e:
                    extracted_text = f"❌ OCR 실행 중 오류: {e}"

                # OCR 결과 표시
                st.subheader("📄 OCR 추출 텍스트:")
                st.code(extracted_text)
                st.session_state.extracted_text = extracted_text
                # GPT 첨삭 결과
                # st.subheader("🤖 GPT 첨삭 결과:")
                if 'grading_criteria' not in st.session_state:
                    st.session_state.grading_criteria = False
                if 'model_answer' not in st.session_state:
                    st.session_state.model_answer = False
                if "❌" not in extracted_text and st.session_state['question_id']:
                    st.session_state.grading_criteria = grader.get_document_content(st.session_state['question_id'], "채점기준")
                    st.session_state.model_answer = grader.get_document_content(st.session_state['question_id'], "모범답안")
                    correction_result = grader.grade_essay(st.session_state['question_id'], extracted_text)
                    
                    display_correction_with_diff(extracted_text, st.session_state.model_answer, correction_result)
                else:
                    st.info("텍스트 추출 결과가 없어 GPT 첨삭을 실행할 수 없습니다.")

    # 챗봇 섹션
    st.markdown("---")
    st.markdown("## 💬 GPT 챗봇과 대화해보세요")
    chat_input = st.text_input("질문을 입력하세요:", key="chat_input")

    if st.button("질문하기", key="chat_button") and chat_input:
        with st.spinner("답변 생성 중..."):
            gpt_response = grader.mento_chat(st.session_state.grading_criteria, st.session_state.model_answer, st.session_state.extracted_text, chat_input, st.session_state.chat_history)
            st.session_state.chat_history.append({"user": chat_input, "assistant": gpt_response})

    for i, turn in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**👤 질문:** {turn['user']}")
        st.markdown(f"**🤖 GPT:**\n{turn['assistant']}")
        st.markdown("---")

    if st.button("🏠 홈으로 돌아가기", key="home_return"):
        st.session_state.page = "home"
        st.rerun()

def main():
    
    # 세션 상태 초기화
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0
    if "page_num" not in st.session_state:
        st.session_state.page_num = 0
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = "문항1"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if st.query_params.get("page"):
        st.session_state.page = st.query_params.get("page")

    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "grading":
        render_grading()
    elif st.session_state.page == "exam":
        render_exam()

if __name__ == "__main__":
    main()
    
    
    
