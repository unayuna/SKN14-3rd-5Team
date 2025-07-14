import os
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 환경변수는 직접 입력하거나 별도 관리 권장
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


def clean_text(text: str) -> str:
    """헤더/페이지번호 삭제"""
    text = re.sub(r"국민취업지원제도 참여자 안내서", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_docs(docs):
    """문서 전처리 및 청크 분할"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=['\n\n', '\n', '.', ' ', '']
    )
    clean_docs = [
        doc.copy(update={'page_content': clean_text(doc.page_content)})
        for doc in docs
    ]
    chunks = splitter.split_documents(clean_docs)
    return clean_docs, chunks


def build_vector_store(docs):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vector_store


def get_summary_chain():
    summary_prompt = ChatPromptTemplate.from_messages([
        ('system', '당신은 정부 지원 공고문을 핵심만 뽑아 알려주는 정확하고 친절한 챗봇입니다.'),
        ('human', """
다음 공고문 내용을 항목별로 요약해 주세요.
* 제공된 context만을 참고하여 요약해 주세요.
* 공고문을 요약할 때 context에서 확인할 수 없는 내용에 대해서 지어내거나 상상해서 이야기 하지 마세요.
* 모르는 내용에 대해서는 '해당 질문은 공고문에서 확인할 수 없습니다. 문의처로 문의 바랍니다.'라고 답변해 주세요.

공고문:
{context}

[최종 응답형식]
- 공고명:\n\n
- 신청대상:\n\n
- 신청기간:\n\n
- 지원내용:\n\n
- 제출서류:\n\n
- 문의처:\n\n
         
        """)
    ])
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    return summary_prompt | llm


def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={'k':3})
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', '당신은 공고문의 내용만 근거로 답변하는 정확하고 친절한 챗봇입니다.'),
        ('human', """
* 제공된 context만을 참고하여 다음 질문에 답변해주세요.
* 공고문을 요약할 때 context에서 확인할 수 없는 내용에 대해서 지어내거나 상상해서 이야기 하지 마세요.
* 모르는 내용에 대해서는 '해당 질문은 공고문에서 확인할 수 없습니다. 문의처로 문의 바랍니다.'라고 답변해 주세요.

[사용자 질문]
{query}

[참고 문서]
{context}

[최종 응답형식]
답변:

        """)
    ])
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    output_parser = StrOutputParser()
    qa_chain = (
        {'context': retriever, 'query': RunnablePassthrough()}
        | qa_prompt | llm | output_parser
    )
    return qa_chain

# Streamlit 등에서 import해서 사용할 수 있도록 클래스화
class NoticeRAG:
    def __init__(self, path: str):
        self.docs = load_pdf(path)
        self.clean_docs, self.chunks = preprocess_docs(self.docs)
        self.vector_store = build_vector_store(self.docs)
        self.summary_chain = get_summary_chain()
        self.qa_chain = get_qa_chain(self.vector_store)

    def summary(self):
        context_text = '\n\n'.join([chunk.page_content for chunk in self.chunks])
        result = self.summary_chain.invoke({'context': context_text})
        return result.content

    def qa(self, user_input: str):
        return self.qa_chain.invoke(user_input)