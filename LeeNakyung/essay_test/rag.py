import os
import json
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv(dotenv_path="../essay_test/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
DATA_DIR = "../essay_test/data"



def load_data():
    all_data = {}
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            qid = data["question_id"]
            year, school, qnum = qid.split("_", 2)
            all_data[qid] = {
                "year": year,
                "school": school,
                "qnum": qnum,
                "data": data,
                "file": fname
            }
    return all_data



def build_vectorstore(question_bank):
    docs = []
    for qid, item in question_bank.items():
        content = f"질문 ID: {qid}\n"
        content += f"[출제 목적]\n{item['data'].get('intended_purpose')}\n\n"
        content += f"[채점 기준]\n{item['data'].get('grading_criteria')}\n\n"
        content += f"[예시 답안]\n{item['data'].get('sample_answer')}"
        doc = Document(page_content=content, metadata={"question_id": qid})
        docs.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
    return FAISS.from_documents(splits, embeddings)


def generate_feedback(llm, question_id, grading_criteria, sample_answer, user_answer, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": {"question_id": question_id}}
    )
    docs = retriever.get_relevant_documents("")
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
[역할]
당신은 대치동에서 10년간 논술을 가르친, 냉철하지만 애정 어린 조언을 아끼지 않는 스타강사 '논리왕 김멘토'입니다.

[입력 정보]
1. [채점 기준]: {grading_criteria}
2. [모범 답안]: {sample_answer}
3. [학생 답안]: {user_answer}

[첨삭 절차 및 지시]
1. (이해) 학생 답안을 전체적으로 읽고 핵심 주장 파악
2. (비교) 채점 기준 및 예시답안과 비교하여 분석
3. (평가) 장단점 명시
4. (종합) 첨삭 문장 완성

[출력 형식]
---
**[총평]**
...

**[잘한 점 (칭찬 포인트) 👍]**
...

**[아쉬운 점 (개선 포인트) ✍️]**
...

**[이렇게 바꿔보세요 (대안 문장 제안) 💡]**
...

**[예상 점수 및 다음 학습 팁 🚀]**
...
"""
    result = llm.invoke(prompt)
    return result.content


def build_answer_chatbot(user_answer):
    doc = Document(page_content=user_answer)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents([doc])
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
    return FAISS.from_documents(splits, embeddings)