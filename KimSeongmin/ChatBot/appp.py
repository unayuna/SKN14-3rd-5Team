import streamlit as st
from PIL import Image
import io
import numpy as np
from paddleocr import PaddleOCR
import fitz  # PDF 미리보기용
from config import UNIVERSITY_DATA
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import cv2

# 페이지 설정 (가장 첫 줄)
st.set_page_config(page_title="GPT 손글씨 첨삭", layout="wide")

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PaddleOCR 초기화
ocr_model = PaddleOCR(use_angle_cls=True, lang='korean')

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

def chat_with_gpt(prompt_text, history=[]):
    messages = [{"role": "system", "content": "넌 손글씨 첨삭 선생님이야. 학생의 글을 친절하게 첨삭하고 논리적 흐름을 보완해줘."}]
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": prompt_text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GPT 호출 중 오류 발생: {e}"

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
    question_keys = list(UNIVERSITY_DATA.get(selected_univ, {}).get(selected_year, {}).get("문항수", {}).keys())
    selected_question = st.selectbox("문항 선택", ["선택"] + question_keys)

    if st.button("📄 보기", key="view_exam") and selected_question != "선택":
        st.session_state.selected_question = selected_question
        pdf_path = UNIVERSITY_DATA[selected_univ][selected_year]["문항수"][selected_question]["pdf"]
        page_list = UNIVERSITY_DATA[selected_univ][selected_year]["문항수"][selected_question]["page"]

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(page_list)
            cur_page = st.session_state.page_num

            st.markdown(f"**페이지 {cur_page + 1} / {total_pages}**")
            page = doc[page_list[cur_page]]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            st.image(pix.tobytes("png"), use_column_width=True)

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
            st.image(image, caption=f"{current_file.name} ({index + 1}/{len(selected_files)})", use_column_width=True)

            with st.expander("🔍 이미지 확대 보기"):
                st.image(image, use_column_width=True)

            if st.button("🤖 GPT 첨삭 실행", key=f"gpt_feedback_{index}"):
                st.markdown("## 📄 첨삭 결과")



                # 이미지 전처리
                image = Image.open(current_file).convert('RGB')
                img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # OCR 수행
                try:
                    result = ocr_model.ocr(img_np)

                    if result and isinstance(result[0], list) and len(result[0]) > 0:
                        # 텍스트 추출
                        extracted_lines = []
                        for line in result[0]:
                            if isinstance(line, list) and len(line) > 1:
                                text = line[1][0]
                                extracted_lines.append(text)
                        extracted_text = "\n".join(extracted_lines)
                    else:
                        extracted_text = "❌ 인식된 텍스트가 없습니다."

                except Exception as e:
                    extracted_text = f"❌ OCR 실행 중 오류: {e}"

                # OCR 결과 표시
                st.subheader("📄 OCR 추출 텍스트:")
                st.code(extracted_text)

                # GPT 첨삭 결과
                st.subheader("🤖 GPT 첨삭 결과:")
                if "❌" not in extracted_text:
                    gpt_feedback = chat_with_gpt(extracted_text, history=[])
                    st.markdown(gpt_feedback)
                else:
                    st.info("텍스트 추출 결과가 없어 GPT 첨삭을 실행할 수 없습니다.")

    # 챗봇 섹션
    st.markdown("---")
    st.markdown("## 💬 GPT 챗봇과 대화해보세요")
    chat_input = st.text_input("질문을 입력하세요:", key="chat_input")

    if st.button("질문하기", key="chat_button") and chat_input:
        gpt_response = chat_with_gpt(chat_input, history=st.session_state.chat_history)
        st.session_state.chat_history.append({"user": chat_input, "assistant": gpt_response})

    for i, turn in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**👤 질문:** {turn['user']}")
        st.markdown(f"**🤖 GPT:** {turn['assistant']}")
        st.markdown("---")

    if st.button("🏠 홈으로 돌아가기", key="home_return"):
        st.session_state.page = "home"
        st.rerun()


if st.query_params.get("page"):
    st.session_state.page = st.query_params.get("page")

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "grading":
    render_grading()
elif st.session_state.page == "exam":
    render_exam()
