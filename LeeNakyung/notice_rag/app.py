import streamlit as st
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 불러오기
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = openai_key

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile

st.title("📚 PDF 기반 RAG Q&A 시스템")

uploaded_file = st.file_uploader("PDF 문서를 업로드하세요", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF 로딩 완료")

    if st.button("🔍 문서 벡터화 시작"):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(split_docs, embedding=embeddings)

        st.session_state.vectordb = vectordb
        st.success("✅ 벡터화 완료")

if "vectordb" in st.session_state:
    user_question = st.text_input("💬 질문을 입력하세요:")

    if user_question:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # 또는 gpt-4
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectordb.as_retriever()
        )

        with st.spinner("답변 생성 중..."):
            answer = qa_chain.run(user_question)

        st.markdown("### 📢 답변:")
        st.write(answer)
