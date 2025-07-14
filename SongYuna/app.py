import streamlit as st
from notice_rag import NoticeRAG

st.set_page_config(page_title='공고문 요약 및 질의응답 RAG', layout='wide')

# 세션 상태 초기화
if "rag" not in st.session_state:
    st.session_state.rag = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# 제목과 설명
st.title("공고문 요약 및 Q&A 챗봇")
st.write("업로드한 공고문을 요약한 후, 질문에 답변합니다.")

# 파일 업로드
uploaded_file = st.file_uploader("공고문 PDF 파일을 업로드하세요.", type=["pdf"])

if uploaded_file:
    with open('uploaded_notice.pdf', 'wb') as f:
        f.write(uploaded_file.read())
    # NoticeRAG 인스턴스 생성
    st.session_state.rag = NoticeRAG("uploaded_notice.pdf")
    st.session_state.summary = None
    st.session_state.history = []
    st.success("파일 업로드 및 분석 완료!")
else:
    st.warning('파일을 업로드 해주세요.')

# 공고문 요약
if st.session_state.rag and st.session_state.summary is None:
    with st.spinner('요약 생성 중...'):
        summary = st.session_state.rag.summary()
        st.session_state.summary = summary

# 요약본 출력 및 질문 입력
if st.session_state.summary:
    st.subheader('공고문 요약')
    st.markdown(st.session_state.summary)

    # 이전 대화 기록 출력
    st.subheader("Q&A")
    for qa in st.session_state.history:
        st.markdown(f'질문: {qa["question"]}')
        st.markdown(f'답변: {qa["answer"]}')

    # 사용자 질문 입력
    st.text_input("질문을 입력하세요:", key="user_input")

    if st.button("확인"):
        query = st.session_state['user_input'].strip()
        if query:
            with st.spinner('답변 생성 중...'):
                answer = st.session_state.rag.qa(query)
                st.write(f"{answer}") 
                if answer:
                    st.session_state.history.append({
                        "question": query,
                        "answer": answer
                    })
                else:
                    st.warning('답변이 생성되지 않았습니다.')
        else:
            st.warning('질문을 입력해주세요.')
