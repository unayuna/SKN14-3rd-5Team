import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec
import time

class Summary_chatbot:
    def __init__(self, pdf_path):
        load_dotenv()
        self.loader = UnstructuredPDFLoader(pdf_path)
        raw_documents = self.loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.documents = text_splitter.split_documents(raw_documents)
        self.embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])
        """ Pinecone
        pc = PineconeClient(api_key = os.getenv("PINECONE_API_KEY"))
        pinecone_index = os.environ['PINECONE_INDEX_NAME']
        
        existing_indexes = [idx['name'] for idx in pc.list_indexes()]
        if pinecone_index  in existing_indexes:
            pc.delete_index(pinecone_index)
            for _ in range(60):
                time.sleep(0.5)
                existing_indexes = [idx['name'] for idx in pc.list_indexes()]
                if pinecone_index not in existing_indexes:
                    break
        
        
        
        if pinecone_index not in existing_indexes:
            pc.create_index(
                name=pinecone_index,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = pc.Index(name = pinecone_index)
        self.vectorstore = PineconeVectorStore(self.index, self.embeddings, 'text')
        if len(self.index.describe_index_stats()['namespaces']) == 0:
            self.vectorstore.add_documents(self.documents)
        """
        # FAISS
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_template(
            """문서 내용을 참고하여 질문에 답변해줘. 답변에 대한 근거가 되는 문서의 페이지를 "위 내용은 '근거가 되는 문서 페이지'에서 확인 가능합니다"라고 꼭 적어줘. 답변의 마지막엔 '자세한 내용은 문서를 확인해주시기 바랍니다.' 라고 해줘.
            만약 답변할 수 있는 내용이 없으면 '해당 질문에 대한 내용이 없습니다.'라고 답해줘.\n\n문서 내용:\n{context}\n\n질문:{question}\n답변:"""
        )
        self.output_parser = StrOutputParser()
        
        
    
    def ask(self, question):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k":3})
        docs = retriever.get_relevant_documents(question)
        
        context = "\n".join([doc.page_content for doc in docs])
        chain = self.prompt | self.llm | self.output_parser
        answer = chain.invoke({"context" : context, "question" : question})
        
        return answer.strip()
    