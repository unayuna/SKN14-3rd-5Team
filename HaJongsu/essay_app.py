import fitz
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
from essay_rag import Essay_chatbot

UNIVERSITY_DATA = {
    "연세대" : {
        "2023" : {
            "pdf" : "yeonsei_2023.pdf",
            "문항수" : {
                "문항1" : [5 ,6, 7],
            }
        }
    },
    "아주대" : {
        "2023" : {
            "pdf" : "ajou_2023.pdf",
            "문항수" : {
                "문항1" : [1, 2],
                "문항2" : [3, 4]
            }
        }
    }
}

# QUESTION_PAGES = {
#     "문항1" : [1, 2],
#     "문항2" : [3, 4]
# }

st.set_page_config(layout="wide")

def show_question_images(doc, pages):
    cols = st.columns(len(pages))
    for i, p in enumerate(pages):
        page = doc[p]
        pix = page.get_pixmap(dpi=250)
        image = Image.open(BytesIO(pix.tobytes("png")))
        cols[i].image(image, caption=f"페이지 {p}", use_container_width=True)

def render_js_timer(timer_id):
    components.html(f"""
        <div id="timer_{timer_id}" style="font-size:24px; font-weight:bold; color:green; margin: 10px 0;"></div>
        <script>
        if (!sessionStorage.getItem('remaining_{timer_id}')) {{
            sessionStorage.setItem('remaining_{timer_id}', 0);
        }}
        var total = parseInt(sessionStorage.getItem('remaining_{timer_id}'));
        if (isNaN(total)) total = 0;

        function updateTimer() {{
            var paused = sessionStorage.getItem('paused_{timer_id}') === 'true';
            var minutes = Math.floor(total / 60);
            var seconds = total % 60;
            if (paused) {{
                document.getElementById("timer_{timer_id}").innerHTML = "⏸ 타이머가 일시 정지되었습니다. 남은 시간: " + minutes + "분 " + (seconds < 10 ? "0" : "") + seconds + "초";
            }} else if (total > 0) {{
                document.getElementById("timer_{timer_id}").innerHTML = "남은 시간: " + minutes + "분 " + (seconds < 10 ? "0" : "") + seconds + "초";
                total -= 1;
                sessionStorage.setItem('remaining_{timer_id}', total);
            }} else if (total <= 0) {{
                document.getElementById("timer_{timer_id}").innerHTML = "⏰ 시간이 종료되었습니다!";
            }}
            setTimeout(updateTimer, 1000);
        }}

        updateTimer();
        </script>
    """, height=60)

def main():
    st.title("📘 대학 논술 자동 첨삭 챗봇")

    with st.sidebar:
        st.header("📚 대학/연도/문항 선택")
        university = st.selectbox("대학을 선택하세요", list(UNIVERSITY_DATA.keys()))
        year = st.selectbox("연도를 선택하세요", list(UNIVERSITY_DATA[university].keys()))
        question_choice = st.selectbox("문항을 선택하세요", list(UNIVERSITY_DATA[university][year]["문항수"].keys()))   

        st.header("⏱️ 타이머 설정 (분)")
        timer_duration = st.number_input("풀이 시간을 설정하세요 (분 단위)", min_value=1, max_value=180, value=30, key="timer_setting")

    pdf_path = UNIVERSITY_DATA[university][year]['pdf']

    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = Essay_chatbot(pdf_path)
    chatbot = st.session_state['chatbot']

    doc = fitz.open(pdf_path)
    
    question_pages = UNIVERSITY_DATA[university][year]["문항수"][question_choice]

    st.subheader("📄 선택한 문항의 문제지")
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
        st.image(uploaded_file, caption=f"{question_choice} 답안 이미지", use_container_width  = False)
        extracted_text = chatbot.extract_text_from_image(uploaded_file)
        st.session_state['user_answers'][question_choice] = extracted_text
    
    if answer_key in st.session_state['user_answers']:
        st.text_area("📝 OCR로 추출된 답안", st.session_state['user_answers'][question_choice])
    
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
    

if __name__ == "__main__":
    load_dotenv()
    main()