import os
import json
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

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

# class AnswerChatRAG:
#     def __init__(self, user_answer, openai_api_key, embedding_model="text-embedding-ada-002"):
#         self.user_answer = user_answer
#         self.openai_api_key = openai_api_key
#         self.embedding_model = embedding_model
#         self.build_vectorstore = self.build_vectorstore()
#         self.retriever = self.build_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#         self.build_chain = self.build_chain()
#
#     # 유저 답변 -> 벡터스토어 변환
#     def build_vectorstore(self):
#         doc = Document(page_content=self.user_answer)
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         splits = text_splitter.split_documents([doc])
#         embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=self.embedding_model)
#         return FAISS.from_documents(splits, embeddings)
#
#     # 벡터스토어 + 프롬프트 + LLM -> 체인 구성
#     def build_chain(self,vectorstore, openai_api_key):
#         retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#
#         # 체인 구성
#         prompt = PromptTemplate.from_template("""
#                     당신은 10년 이상 수능 및 대학 논술을 전문적으로 가르쳐온 첨삭 전문가입니다.
#                     학생의 질문에 대해 학생이 작성한 논술 문장을 바탕으로 명확하고 구체적인 피드백을 제공합니다.
#
#                     [제시 문장]
#                     아래는 벡터 검색을 통해 선택된 학생의 답안 내용 일부입니다. 참고해 분석에 활용하세요.
#
#                     {context}
#
#                     [학생 질문]
#                     {question}
#
#                     [답변 지침]
#                     1. 질문의 요지를 파악하고, 답안 문장 중 관련 있는 내용을 연결해 해석합니다.
#                     2. 부족하거나 개선이 필요한 부분이 있다면 논리적으로 설명하고 구체적인 문장 또는 방향을 제안합니다.
#                     3. 피드백은 친절하고 조리 있게 제시하되, 논리성과 구조적 사고력을 기를 수 있도록 유도합니다.
#                     4. 학생이 잘 이해할 수 있도록 길고 구체적으로, 상세히 답변해줍니다.
#
#                     [답변 형식 예시]
#                     ### 🧠 분석
#                     - (질문 요지를 요약하고, 학생 답안에서 관련 문장을 어떻게 해석했는지 설명)
#
#                     ### 💡 개선 제안
#                     - (보다 나은 문장 표현 / 논리 전개 / 사례 추가 등 구체적 개선 방법 제안)
#
#                     ### 🗒️ 예시 답변
#                     - (분석과 개선 제안을 토대로 모범 답안 혹은 진행 방향을 예시로 보여주기)
#
#                     ### 🏁 요약 및 다음 단계
#                     - (종합 정리와 향후 유사 질문 대비 학습 팁)
#
#                     [답변]
#                     """)
#
#         chain = (
#                 {
#                     "context": lambda x: "\n\n".join([
#                         doc.page_content for doc in retriever.get_relevant_documents(x["question"])
#                     ]),
#                     "question_id": lambda x: x["question_id"]
#                 }
#                 | prompt
#                 | ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
#                 | StrOutputParser()
#         )
#         return chain
#
#     def invoke(self, question_id):
#         return self.build_chain.invoke({"question_id": question_id})


class AnswerChatRAG:
    def __init__(self, user_answer, openai_api_key, embedding_model="text-embedding-ada-002"):
        self.user_answer = user_answer
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model

        self.vectorstore = self.build_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        self.chain = self.build_chain()

    # 답변을 벡터스토어로 전환
    def build_vectorstore(self):
        doc = Document(page_content=self.user_answer)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents([doc])
        embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key, model=self.embedding_model
        )
        return FAISS.from_documents(splits, embeddings)

    # 체인 구성 (프롬프트 + LLM + 출력 파서)
    def build_chain(self):
        prompt = PromptTemplate.from_template("""
당신은 10년 이상 수능 및 대학 논술을 전문적으로 가르쳐온 첨삭 전문가입니다.
학생의 질문에 대해 학생이 작성한 논술 문장을 바탕으로 명확하고 구체적인 피드백을 제공합니다.

[제시 문장]
아래는 벡터 검색을 통해 선택된 학생의 답안 내용 일부입니다. 참고해 분석에 활용하세요.

{context}

[학생 질문]
{question}

[답변 지침]
1. 질문의 요지를 파악하고, 답안 문장 중 관련 있는 내용을 연결해 해석합니다.
2. 부족하거나 개선이 필요한 부분이 있다면 논리적으로 설명하고 구체적인 문장 또는 방향을 제안합니다.
3. 피드백은 친절하고 조리 있게 제시하되, 논리성과 구조적 사고력을 기를 수 있도록 유도합니다.
4. 학생이 잘 이해할 수 있도록 길고 구체적으로, 상세히 답변해줍니다.

[답변 형식 예시]
### 🧠 분석
- (질문 요지를 요약하고, 학생 답안에서 관련 문장을 어떻게 해석했는지 설명)

### 💡 개선 제안
- (보다 나은 문장 표현 / 논리 전개 / 사례 추가 등 구체적 개선 방법 제안)

### 🗒️ 예시 답변
- (분석과 개선 제안을 토대로 모범 답안 혹은 진행 방향을 예시로 보여주기)

### 🏁 요약 및 다음 단계
- (종합 정리와 향후 유사 질문 대비 학습 팁)

[답변]
""")
        chain = (
            {
                "context": lambda x: "\n\n".join([
                    doc.page_content for doc in self.retriever.get_relevant_documents(x["question"])
                ]),
                "question": lambda x: x["question"]
            }
            | prompt
            | ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=self.openai_api_key)
            | StrOutputParser()
        )
        return chain

    # 외부에서 체인 호출
    def invoke(self, question: str) -> str:
        return self.chain.invoke({"question": question})