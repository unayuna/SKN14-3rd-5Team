import os
from PIL import Image
from dotenv import load_dotenv
import pytesseract

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class Essay_chatbot:
    def __init__(self, pdf_path):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])
        self.vectorstore = self.build_vectorstore_from_pdf(pdf_path)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.retriever = self.vectorstore.as_retriever(search_type='similarity', search_kwargs={"k":3})
        
    def extract_text_from_image(self, image_file):
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image, lang="kor+eng")

        return text
    
    def build_vectorstore_from_pdf(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        return vectorstore
    
    def feedback_rag(self, user_text):

        qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            retriever=self.retriever,
            return_source_documents=False,
        )

        prompt = (
            f"다음은 학생의 손글씨 답안을 OCR로 추출한 텍스트입니다.:\n\n{user_text}\n\n"
            "첨삭 기준은 논술 모범답안에 따라 정확성, 논리성, 표현력을 중심으로 평가해 주세요. "
            "모범답안을 참고하여 학생의 답안에 대해 칭찬과 개선점을 모두 포함해 첨삭 피드백을 작성해주세요."
        )

        return qa_chain.run(prompt)
