from PIL import Image
import pytesseract
from paddleocr import PaddleOCR
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class EssayChatbot:
    def __init__(self, pdf_path):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])
        # self.vectorstore = self.build_vectorstore_from_pdf(pdf_path)
        faiss_path = pdf_path.split('.')[0]
        self.vectorstore = FAISS.load_local(faiss_path, self.embeddings)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.retriever = self.vectorstore.as_retriever(search_type='similarity')
        self.ocr = PaddleOCR(use_angle_cls=True, lang='korean')

    def build_vectorstore_from_pdf(self, pdf_path):
        documents = PyMuPDFLoader(pdf_path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
        return FAISS.from_documents(chunks, self.embeddings)

    def extract_text_from_image_tesseract(self, image_file):
        print("Extracting text from image...")
        pil_image = Image.open(image_file)
        # preprocessed_image  = self.preprocess_image(pil_image)
        custom_config = r'--oem 3 --psm 3 -l kor'
        # return pytesseract.image_to_string(preprocessed_image , config=custom_config)
        return pytesseract.image_to_string(pil_image, config=custom_config)
    
    def extract_text_from_image_paddle(self, image_file):
        print("Extracting text from image...")
        img = np.array(Image.open(image_file).convert("RGB"))
        result = self.ocr.ocr(img)
        ocr_result = result[0]
        # extracted_text = "\n".join([line[1][0] for line in result[0]])
        extracted_text = "\n".join([line for line in ocr_result['rec_texts']])
        return extracted_text

    # def feedback_rag(self, user_text):
    #     qa_chain = RetrievalQA.from_chain_type(
    #         llm=self.llm,
    #         retriever=self.retriever,
    #         return_source_documents=False,
    #     )
    #     prompt = (
    #         f"다음은 학생의 손글씨 답안을 OCR로 추출한 텍스트입니다.:\n\n{user_text}\n\n"
    #         "첨삭 기준은 논술 모범답안에 따라 정확성, 논리성, 표현력을 중심으로 평가해 주세요. "
    #         "모범답안을 참고하여 학생의 답안에 대해 칭찬과 개선점을 모두 포함해 첨삭 피드백을 작성해주세요."
    #     )
    #     return qa_chain.run(prompt)

    def feedback_rag(self, user_text, filter_metadata):
        docs = self.retriever.get_relevant_documents(user_text, k=5)

        if filter_metadata:
            docs = [
                d for d in docs
                if all(d.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
            
        if not docs:
            return "❗ 유사한 문항 정보를 찾을 수 없습니다."
        
        context = docs[0].page_content

        prompt = (
            f"[모범답안, 출제의도, 채점기준]\n{context}\n\n"
            f"[학생 답안]\n{user_text.strip()}\n\n"
            "위의 채점기준과 모범답안을 바탕으로 학생의 답안을 정성스럽게 첨삭하고, "
            "구체적인 고득점 전략도 함께 제시해주세요."
        )

        return self.llm.invoke(prompt)
    
    def preprocess_image(self, pil_image):
        image = np.array(pil_image.convert("L"))  # Grayscale
        image = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )
        return Image.fromarray(image)
