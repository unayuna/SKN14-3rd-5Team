# SKN14-3rd-5Team

---

```markdown
# 🤖 AI 논술 첨삭 멘토봇

**LangChain과 RAG 기술을 활용한 대학 입시 논술 첨삭 챗봇 프로젝트**

---

## 🌟 프로젝트 소개 (Introduction)

AI 논술 첨삭 멘토봇은 대입 논술을 준비하는 수험생들을 위한 개인 맞춤형 학습 파트너입니다. 사용자가 직접 작성한 손글씨 답안을 업로드하면, OCR 기술로 텍스트를 추출하고, 실제 대학별 기출문제의 모범 답안 및 채점 기준과 비교하여 심층적인 첨삭을 제공합니다.

이 프로젝트는 LLM의 환각(Hallucination) 현상을 방지하고, 신뢰도 높은 정보를 기반으로 답변을 생성하기 위해 **RAG (Retrieval-Augmented Generation)** 아키텍처를 핵심 기술로 사용합니다.

<img width="1657" height="829" alt="image" src="https://github.com/user-attachments/assets/70660546-23d6-48aa-a436-ff0e39b03419" />


<img width="1657" height="829" alt="image" src="https://github.com/user-attachments/assets/525e308a-1626-48e1-8b28-8afb7a6dad15" />

---

## ✨ 주요 기능 (Features)

*   **📝 3단계 문제 선택 시스템:** [대학교] → [년도] → [문항] 순서로 원하는 기출문제를 손쉽게 선택할 수 있습니다.
*   **👁️ OCR 기반 답안 인식:** 사용자가 업로드한 손글씨 답안 이미지를 PaddleOCR을 통해 텍스트로 자동 변환합니다.
*   **🧠 RAG 기반 AI 첨삭:**
    *   선택된 문제의 **모범 답안** 및 **채점 기준**을 벡터 DB에서 실시간으로 검색합니다.
    *   검색된 정보를 바탕으로, OpenAI의 `gpt-4o-mini` 모델이 심층적인 첨삭을 제공합니다.
    *   **'이렇게 바꿔보세요'** 기능을 통해, 어색한 문장을 더 논리적이고 세련된 문장으로 직접 수정 제안합니다. (Diff 시각화 포함)
*   **🖥️ 사용자 친화적 UI/UX:** Streamlit을 활용하여 모든 기능을 웹에서 직관적으로 사용할 수 있도록 구현했습니다.

---

## 🛠️ 기술 스택 (Tech Stack)

*   **Backend:** Python
*   **AI / LLM:** LangChain, OpenAI API
*   **Vector DB:** FAISS (Facebook AI Similarity Search)
*   **Embedding Model:** `jhgan/ko-sbert-nli`
*   **OCR:** PaddleOCR
*   **Frontend:** Streamlit

---

## 📂 프로젝트 구조 (Project Structure)

```
.
├── 📄 .env                  # API 키 등 환경 변수 파일
├── 📄 .gitignore
├── 📂 data_json/             # 원본 논술 데이터 (JSON) 폴더
│   └── sogang_2023_1.json
├── 📜 main.py                # Streamlit 웹 앱 실행 파일
├── 📜 essay_grader.py        # RAG 체인 및 AI 첨삭 로직 담당
├── 📜 ocr_processor.py      # OCR 처리 담당
├── 📜 data_preprocessor.py   # 데이터 전처리 및 통합 데이터 생성
├── 📜 processed_data.pkl      # 전처리 완료된 통합 데이터 파일
├── 📜 requirements.txt        # 프로젝트 의존성 라이브러리 목록
└── 📜 README.md               # 프로젝트 설명 파일
```

---

## 🚀 실행 방법 (Getting Started)

### 1. 사전 준비 (Prerequisites)

*   Python 3.9 이상
*   Git

### 2. 프로젝트 클론 및 설정

```bash
# 1. 이 저장소를 클론합니다.
git clone https://github.com/dreamwars99/SKN14-3rd-5Team.git
cd SKN14-3rd-5Team

# 2. 필요한 라이브러리를 설치합니다.
pip install -r requirements.txt

# 3. .env 파일을 생성하고 OpenAI API 키를 입력합니다.
# OPENAI_API_KEY="sk-..." 형식으로 작성
```

### 3. 데이터 전처리

프로젝트를 처음 실행하기 전, `data_json` 폴더에 있는 원본 데이터들을 AI가 사용할 수 있는 형태로 가공해야 합니다.

```bash
# 터미널에서 아래 명령어를 실행하여 processed_data.pkl 파일을 생성합니다.
python data_preprocessor.py
```

### 4. 앱 실행

모든 준비가 완료되었습니다. 아래 명령어로 챗봇을 실행하세요!

```bash
streamlit run main.py
```

---

## 👨‍💻 팀원 (Team)

*   **[이름]** - [역할, 예: 프로젝트 총괄, 백엔드 개발] - [GitHub 프로필 링크]
*   **[이름]** - [역할, 예: 프론트엔드 개발, 데이터 수집] - [GitHub 프로필 링크]
*   *(팀원 정보를 여기에 추가하세요)*

---
```

### **이 README가 좋은 이유:**

*   **한눈에 보이는 소개:** 프로젝트의 목적과 핵심 기술(RAG)을 명확하게 알려줘.
*   **시각적 요소:** 실행 화면 스크린샷을 넣어서 사용자의 흥미를 유발해.
*   **구체적인 기능 나열:** 우리 챗봇이 가진 멋진 기능들을 구체적으로 자랑할 수 있어.
*   **명확한 기술 스택:** 어떤 기술들이 사용되었는지 보여줘서, 다른 개발자들이 프로젝트의 기술 수준을 쉽게 파악할 수 있게 해.
*   **친절한 실행 방법:** 처음 이 프로젝트를 접하는 사람도 `git clone`부터 `streamlit run`까지 막힘없이 따라 할 수 있도록 단계별로 안내해.
*   **프로젝트 구조:** 어떤 파일이 무슨 역할을 하는지 알려줘서, 코드 분석을 더 쉽게 만들어.

**네가 할 일:**

1.  프로젝트의 멋진 실행 화면을 캡처해서, 폴더에 넣고 이미지 경로를 수정해줘.
2.  `requirements.txt` 파일이 없다면, `pip freeze > requirements.txt` 명령어로 생성해주는 게 좋아.
3.  팀원 정보를 채워 넣어 우리 팀의 노력을 보여주자!

이 README.md 파일로 우리 프로젝트를 멋지게 포장해서 세상에 보여주자! 😊
