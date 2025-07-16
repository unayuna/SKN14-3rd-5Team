import streamlit as st
from dotenv import load_dotenv
from core.chatbot import EssayChatbot
from config import UNIVERSITY_DATA
from ui import show_question_images, render_js_timer
import streamlit.components.v1 as components
import fitz

load_dotenv()
st.set_page_config(layout="wide")

def main():
    
    st.title("📘 대학 논술 자동 첨삭 챗봇")

    with st.sidebar:
        st.header("📚 대학/연도/문항 선택")
        university = st.selectbox("대학 선택", list(UNIVERSITY_DATA.keys()))
        year = st.selectbox("연도 선택", list(UNIVERSITY_DATA[university].keys()))
        question_choice = st.selectbox("문항 선택", list(UNIVERSITY_DATA[university][year]["문항수"].keys()))
    
        st.header("⏱️ 타이머 설정 (분)")
        timer_duration = st.number_input("⏱ 풀이 시간 설정 (분)", min_value=1, max_value=180, value=30)

    pdf_path = UNIVERSITY_DATA[university][year]["pdf"]
    question_pages = UNIVERSITY_DATA[university][year]["문항수"][question_choice]

    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = EssayChatbot(pdf_path)
    chatbot = st.session_state['chatbot']

    st.subheader("📄 문제 이미지")
    doc = fitz.open(pdf_path)
    show_question_images(doc, question_pages)

    # 타이머 상태 키 생성
    timer_key = f"timer_active_{question_choice}"
    if timer_key not in st.session_state:
        st.session_state[timer_key] = {
            "running": False,
            "paused": False,
            "seconds": timer_duration * 60
        }
    state = st.session_state[timer_key]
    
    col1, col2, col3 = st.columns([1,1,1])
    with col1:    
        if st.button("▶️ 타이머 시작", key=f"start_timer_{question_choice}"):
            state["running"] = True
            state["paused"] = False
            components.html(f"<script>sessionStorage.setItem('remaining_{question_choice}', {state['seconds']}); sessionStorage.setItem('paused_{question_choice}', 'false');</script>", height=0)

    with col2:
        if st.button("⏯ 일시정지 / 재개", key=f"pause_{question_choice}"):
            if state["running"]:
                state["paused"] = not state["paused"]
                js_pause = 'true' if state['paused'] else 'false'
                components.html(f"<script>sessionStorage.setItem('paused_{question_choice}', '{js_pause}');</script>", height=0)

    with col3:
        if st.button("⏹ 타이머 종료", key=f"stop_{question_choice}"):
            state["running"] = False
            state["paused"] = False
            state["seconds"] = timer_duration * 60
            components.html(f"<script>sessionStorage.removeItem('remaining_{question_choice}'); sessionStorage.setItem('paused_{question_choice}', 'false');</script>", height=0)
    
    if state["running"]:
        render_js_timer(question_choice)
    elif state["paused"]:
        st.info("⏸ 타이머 일시정지 상태입니다 (시간 유지)")

    st.divider()
    st.header("🖋 손글씨 답안 이미지 업로드")

    uploaded_file = st.file_uploader("이미지 업로드 (jpg/png)", type=['jpg', 'png', 'jpeg'])

    if 'user_answers' not in st.session_state:
        st.session_state['user_answers'] = {}
    
    answer_key = f"{university}_{year}_{question_choice}"

    if uploaded_file:
        with st.spinner("OCR로 답안 추출 중..."):
            st.image(uploaded_file, caption=f"{question_choice} 답안 이미지", use_container_width  = False)
            # extracted_text = chatbot.extract_text_from_image_tesseract(uploaded_file)
            extracted_text = chatbot.extract_text_from_image_paddle(uploaded_file)
            st.session_state['user_answers'][answer_key] = extracted_text
    
    if answer_key in st.session_state['user_answers']:
        st.text_area("📝 OCR로 추출된 답안", st.session_state['user_answers'][answer_key])
    
    feedback_key = f"feedback_{question_choice}"
    submit_key = f"submit_{question_choice}"

    if st.button("✍️ GPT로 첨삭하기", key=submit_key):
        with st.spinner("GPT가 첨삭 중입니다..."):
            feedback = chatbot.feedback_rag(st.session_state['user_answers'].get(question_choice, ""))
            st.session_state[feedback_key] = feedback
    
    feedback = st.session_state.get(feedback_key, None)

    if feedback:    
        st.markdown("### ✅ GPT 첨삭 결과")
        st.write(feedback)

if __name__ == '__main__':
    main()
