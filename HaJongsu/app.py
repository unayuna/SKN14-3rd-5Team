import streamlit as st
import os
from rag_test import Summary_chatbot

st.set_page_config(page_title="PDF RAG 챗봇", layout='wide')

st.title("📄 PDF 기반 RAG 챗봇")
st.write("PDF 문서 내용을 기반으로 질문에 답변합니다.")

uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])

if uploaded_file:
    temp_path = 'temp_uploaded.pdf'
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # 파일이 바뀌면 챗봇 새로 생성
    file_hash = hash(uploaded_file.getvalue())
    if st.session_state.get("file_hash") != file_hash:
        st.session_state["file_hash"] = file_hash
        st.session_state.pop("chatbot", None)
        st.session_state.pop("messages", None)
    
    if "chatbot" not in st.session_state:
        with st.spinner("문서를 읽고 인덱싱 중입니다. (수십 초 소요될 수 있습니다)"):
            st.session_state["chatbot"] = Summary_chatbot(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
    
    user_input = st.chat_input("질문을 입력하세요.")
    if user_input:
        st.session_state["messages"].append({"role":"user", "content": user_input})
        st.chat_message("user").write(user_input)
        with st.spinner("답변 생성 중..."):
            answer = st.session_state["chatbot"].ask(user_input)
        st.session_state["messages"].append({"role":"assistant", "content":answer})
        st.chat_message("assistant").write(answer)
else:
    st.info("PDF 파일을 업로드해주세요.")
    st.session_state.pop("chatbot", None)
    st.session_state.pop("messages", None)