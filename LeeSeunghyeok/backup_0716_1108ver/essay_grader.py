# essay_grader.py

# --- 필요한 라이브러리 임포트 ---
import os
from dotenv import load_dotenv
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# --- 논술 첨삭을 담당하는 핵심 클래스 ---
class EssayGrader:
    """
    논술 시험 자료를 로드하여 벡터 DB를 구축하고,
    RAG 체인을 통해 학생 답안을 첨삭하는 클래스.
    """
    def __init__(self, json_path: str):
        print("논술 첨삭기 초기화를 시작합니다...")
        self._setup_api_key()
        self.embedding_model = self._initialize_embedding_model()
        structured_docs = self._load_and_structure_data(json_path)
        print("\n문서 조각들을 벡터로 변환하여 DB에 저장합니다...")
        self.vector_db = FAISS.from_documents(structured_docs, self.embedding_model)
        self.retriever = self.vector_db.as_retriever()
        print("✅ 벡터 데이터베이스 구축 완료!")
        self.correction_chain = self._build_rag_chain()
        print("✅ AI 논술 첨삭 RAG 체인 완성!")
        print("\n--- 모든 준비 완료! 이제 첨삭을 시작할 수 있습니다. ---")

    def _setup_api_key(self):
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("환경변수에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        print("✅ API 키 로딩 완료.")

    def _initialize_embedding_model(self):
        print("임베딩 모델을 로딩합니다... (시간이 좀 걸릴 수 있어요)")
        model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        print("✅ 임베딩 모델 로딩 완료.")
        return model

    def _load_and_structure_data(self, json_path: str):
        print(f"논술 자료 '{json_path}' 파일을 로딩합니다.")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"'{json_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        structured_docs = []
        base_metadata = {
            "university": data.get("university", "정보 없음"),
            "year": data.get("year", "정보 없음"),
            "subject": data.get("subject", "정보 없음")
        }
        content_map = {
            "출제의도": data.get("intended_purpose"),
            "채점기준": data.get("grading_criteria"),
            "모범답안": data.get("sample_answer")
        }
        for content_type, content in content_map.items():
            if content:
                doc = Document(
                    page_content=content,
                    metadata={**base_metadata, "content_type": content_type, "question_id": data.get("question_id")}
                )
                structured_docs.append(doc)
        print(f"✅ 총 {len(structured_docs)}개의 논리적 문서 조각 생성 완료.")
        return structured_docs

    def _build_rag_chain(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        output_parser = StrOutputParser()
        prompt_template = """
        [역할]
        당신은 대치동에서 10년간 논술을 가르친, 냉철하지만 애정 어린 조언을 아끼지 않는 스타강사 '논리왕 김멘토'입니다. 학생의 눈높이에 맞춰 핵심을 꿰뚫는 '팩트 폭격'과 따뜻한 격려를 겸비한 첨삭 스타일로 유명합니다.
        [입력 정보]
        1. [채점 기준]: {retrieved_scoring_criteria}
        2. [모범 답안]: {retrieved_model_answer}
        3. [학생 답안]: {user_ocr_answer}
        [첨삭 절차 및 지시]
        당신은 아래 4단계의 사고 과정을 거쳐, 최종 첨삭문을 [출력 형식]에 맞춰 생성해야 합니다.
        1. (이해): 먼저, [학생 답안]을 한 문단씩 읽으며 전체적인 논리의 흐름과 핵심 주장을 파악합니다.
        2. (비교): 그 다음, 학생 답안의 각 문단이 [채점 기준]의 어떤 항목에 부합하는지, 그리고 [모범 답안]의 논리 구조와 어떻게 다른지 비교 분석합니다.
        3. (평가): 분석한 내용을 바탕으로, 각 항목별로 구체적인 칭찬과 개선점을 정리합니다.
        4. (종합): 마지막으로, 이 모든 내용을 종합하여 아래 [출력 형식]에 맞춰 최종 첨삭문을 작성합니다.
        [출력 형식]
        ---
        **[총평]**
        (학생 답안의 전반적인 강점과 약점을 2~3문장으로 날카롭게 요약)
        **[잘한 점 (칭찬 포인트) 👍]**
        - (채점 기준과 비교하여, 학생 답안이 어떤 점에서 훌륭한지 구체적인 근거와 문장을 인용하여 칭찬)
        **[아쉬운 점 (개선 포인트) ✍️]**
        - (모범답안과 비교하여, 어떤 부분을 보완하면 더 좋은 글이 될 수 있을지 구체적으로 제안)
        **[이렇게 바꿔보세요 (대안 문장 제안) 💡]**
        - (아쉬운 점으로 지적된 문장 1~2개를, 더 논리적이고 세련된 문장으로 직접 수정해서 제안합니다. 반드시 "학생 원문: ..." -> "수정 제안: ..." 형식을 정확히 지켜서 여러 개를 작성해주세요.)
        **[예상 점수 및 다음 학습 팁 🚀]**
        - (채점 기준을 근거로 예상 점수를 제시하고, 이 학생이 다음번에 더 성장하기 위한 구체적인 학습 팁을 1가지 제안)
        ---
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = (
            {
                "retrieved_model_answer": RunnableLambda(lambda x: self.retriever.invoke(f"{x['question_info']} 모범답안")),
                "retrieved_scoring_criteria": RunnableLambda(lambda x: self.retriever.invoke(f"{x['question_info']} 채점기준")),
                "user_ocr_answer": lambda x: x["user_ocr_answer"]
            }
            | prompt
            | llm
            | output_parser
        )
        return chain

    def grade_essay(self, question_info: str, student_answer: str) -> str:
        print(f"'{question_info}'에 대한 첨삭을 시작합니다...")
        return self.correction_chain.invoke({
            "question_info": question_info,
            "user_ocr_answer": student_answer
        })

    def get_model_answer(self, question_info: str) -> str:
        model_answer_docs = self.retriever.invoke(f"{question_info} 모범답안")
        return model_answer_docs[0].page_content if model_answer_docs else "모범답안을 찾을 수 없습니다."