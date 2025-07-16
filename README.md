# SKN14-3rd-5Team

# 🤖 AI 논술 첨삭 멘토봇

**"학원에 가지 않고, 나만의 AI 멘토와 함께 대입 논술을 정복하세요."**

LangChain과 RAG 기술을 활용하여 개발한 개인 맞춤형 대학 입시 논술 과외 챗봇입니다.

---

## 👨‍💻 팀원 소개 및 역할

| 이름 | 역할 분담 | GitHub |
| :--: | :--- | :---: |
| **[하종수]** | 프로젝트 총괄 및 데이터 수집 및 전처리 | [GitHub](<링크>) |
| **[김성민]** | 프론트엔드 개발 (Streamlit), 데이터 수집 및 전처리 | [GitHub](<링크>) |
| **[송유나]** | 데이터 전처리 및 백엔드 개발 (RAG) | [GitHub](<링크>) |
| **[이나경]** | 데이터 전처리 및 백엔드 개발 (RAG) | [GitHub](<링크>) |
| **[이승혁]** | 백엔드 개발(RAG) 프롬프팅 | [GitHub](<링크>) |

---

## 🎯 프로젝트 주제

**개인 맞춤형 AI 논술 과외 튜터 개발**

---

## 💡 주제 선정 이유 및 배경

### "논술, 꼭 학원에 가야만 할까?"

대입 논술 전형은 많은 수험생에게 중요한 기회이지만, 동시에 큰 부담으로 다가옵니다. 저희는 다음과 같은 문제의식에서 이 프로젝트를 시작했습니다.

1.  **독학의 어려움:** 논술은 정해진 답이 없어 혼자 공부하기 막막합니다. 내가 쓴 글이 잘된 글인지, 어떤 점을 개선해야 하는지 객관적인 피드백을 받기 어렵습니다.
2.  **사교육 의존성:** 결국 많은 학생들이 비싼 비용을 지불하고 논술 학원에 의존하게 됩니다.
3.  **챗봇 시대의 새로운 가능성:** ChatGPT와 같은 AI 챗봇이 활성화된 지금, '개인 과외'의 경험을 기술로 구현할 수 있지 않을까?

이러한 고민 끝에, **학원을 대신할 수 있는 나만의 AI 논술 멘토**를 만들어보자는 목표를 세웠습니다. 저희 챗봇은 언제 어디서든, 실제 기출문제를 바탕으로 깊이 있는 첨삭을 제공하여 교육 격차를 해소하고 모든 수험생에게 공정한 학습 기회를 제공하고자 합니다.

---

## 📊 데이터 (Data)

### 1. 데이터 수집

- **출처:** 각 대학교 입학처 홈페이지에서 공식적으로 제공하는 **논술 가이드북**을 활용했습니다.
- **내용:** 공신력 있는 실제 기출문제, 출제의도, 채점기준, 모범답안 데이터를 수집했습니다.
- ⚠️ **주의:** 수집된 모든 자료의 저작권은 각 대학교에 있으며, **교육 및 연구 목적으로만 사용**되었고 상업적 이용은 절대 불가합니다.

### 2. 데이터 정제 과정

수집한 원본 PDF는 문제와 해설이 혼합되어 있어, AI가 학습하기 좋은 형태로 가공하는 과정이 필요했습니다.

- **문제 원문 추출:** 사용자가 웹 화면에서 문제를 먼저 확인할 수 있도록, 원본 PDF에서 **문제 부분만 별도로 추출**하여 관리했습니다. (현재 버전에서는 '채점 기준'으로 대체)
- **핵심 정보 구조화 (JSON):** AI가 RAG(검색 증강 생성)의 Retriever로 활용할 핵심 정보들(**출제의도, 채점기준, 모범답안**)은 `question_id`를 기준으로 구조화된 **JSON 파일**로 생성했습니다. 이 JSON 데이터는 추후 LangChain의 `Document` 객체로 변환되어 벡터 DB에 저장됩니다.

    또한 streamlit 프런트엔드 환경에서 문제를 보여주기 위해 문제는 pdf로 따로 정리했습니다.

---

## 🛠️ 기술 스택 (Tech Stack)

| **분야**              | **기술 및 라이브러리**                                                                                                                                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🖥️ 프로그래밍 언어 & 개발환경 | <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white" /> <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=VisualStudioCode&logoColor=white" />            |
| 🔗 LLM 체인 및 워크플로우   | <img src="https://img.shields.io/badge/LangChain-005F73?style=for-the-badge&logo=LangChain&logoColor=white" /> <img src="https://img.shields.io/badge/LangGraph-000000?style=for-the-badge&logo=LangChain&logoColor=white" />             |
| 🧠 LLM 모델           | <img src="https://img.shields.io/badge/OpenAI%20GPT--4.1-412991?style=for-the-badge&logo=OpenAI&logoColor=white" /> <img src="https://img.shields.io/badge/OpenAI%20Embeddings-10A37F?style=for-the-badge&logo=OpenAI&logoColor=white" /> |
| 📄 문서 로딩 및 전처리      | <img src="https://img.shields.io/badge/PyMuPDF-00599C?style=for-the-badge&logo=AdobeAcrobatReader&logoColor=white" /> <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />            |
| 📦 벡터 저장소 및 임베딩     | <img src="https://img.shields.io/badge/ChromaDB-FF6F61?style=for-the-badge&logo=Chroma&logoColor=white" />                                                                                                                                |
| 🌐 데이터 수집           | <img src="https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=Selenium&logoColor=white" /> <img src="https://img.shields.io/badge/requests-7A88CF?style=for-the-badge&logo=Python&logoColor=white" />                   |
| 🤖 챗봇 인터페이스         | <img src="https://img.shields.io/badge/Chainlit-FFCC00?style=for-the-badge&logo=Lightning&logoColor=black" />                                                                                                                             |
| 🔐 환경 변수 및 설정 관리    | <img src="https://img.shields.io/badge/python_dotenv-000000?style=for-the-badge&logo=Python&logoColor=white" />                                                                                                                           |
| 💬 메시지 및 커뮤니케이션     | <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=Discord&logoColor=white" />                                                                                                                                |
| 📁 협업 및 형상 관리       | <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white" /> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white" />   


| 구분 | 기술 |
| :--- | :--- |
| **Backend** | Python |
| **AI / LLM** | LangChain, OpenAI API (`gpt-4o-mini`) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Embedding** | `jhgan/ko-sbert-nli` (HuggingFace) |
| **OCR** | PaddleOCR |
| **Frontend** | Streamlit |

---

## 🌊 데이터 흐름도 (Data Flow)

![Data Flow Diagram](<https://github.com/dreamwars99/SKN14-3rd-5Team/blob/main/LeeSeunghyeok/images/Group%2016%20(1).png?raw=true>)


---

## 🚶 사용자 흐름도 (User Flow)

![User Flow Diagram](<https://github.com/dreamwars99/SKN14-3rd-5Team/blob/main/LeeSeunghyeok/images/Group%201%20(1).png?raw=true>)


---

## ✨ 결과물 (Screenshot)

### 메인 화면 및 문제 선택
![메인 화면 스크린샷](<여기에_메인화면_스크린샷_경로.png>)

### AI 첨삭 결과
![첨삭 결과 스크린샷](<여기에_첨삭결과_스크린샷_경로.png>)

---

## 🌱 향후 발전 가능성 (Future Work)

- **[첨삭 히스토리 & 성장 대시보드]**
  - 사용자의 모든 첨삭 기록을 DB에 저장하고, 시간 경과에 따른 예상 점수 변화를 그래프로 시각화합니다.
  - 자주 틀리는 약점 키워드를 분석하여 태그 클라우드로 제공합니다.
- **[문제 추천 시스템]**
  - 사용자의 약점을 분석하여, 이를 보완하는 데 도움이 될 만한 다른 대학의 유사 유형 문제를 자동으로 추천합니다.

---

## 📝 한 줄 회고록

*   **[하종수]:** "수많은 오류와 싸우며 문제 해결 능력을 기를 수 있었던, 값진 디버깅의 여정이었습니다."

*   **[김성민]:** "RAG 아키텍처를 직접 구현해보며 LLM의 한계와 가능성을 동시에 느낄 수 있었습니다."

*   **[송유나]:** "RAG 아키텍처를 직접 구현해보며 LLM의 한계와 가능성을 동시에 느낄 수 있었습니다."

*   **[이나경]:** "RAG 아키텍처를 직접 구현해보며 LLM의 한계와 가능성을 동시에 느낄 수 있었습니다."

*   **[이승혁]:** "RAG 아키텍처를 직접 구현해보며 LLM의 한계와 가능성을 동시에 느낄 수 있었습니다."
