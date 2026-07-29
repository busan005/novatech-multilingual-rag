# 🌐 NovaTech 다국어 CS 상담 에이전트

**Claude Code & LangChain/RAG 기반 다국어 실용 서비스 개발 프로젝트**

이름: [학생 이름을 입력하세요]
학번: [학번을 입력하세요]

## 프로젝트 요약

한국 전자제품 브랜드(가상)의 반품/보증/배송/제품사용/계정 관련 FAQ를 한국어·영어·일본어·중국어
4개 언어로 구축하고, RAG(검색 증강 생성) 파이프라인으로 연동한 다국어 CS 챗봇입니다.

- 사용자가 어떤 언어로 질문해도 **다국어 임베딩**으로 의미 기반 검색을 수행합니다.
- 검색된 FAQ가 질문과 다른 언어여도 **Claude**가 맥락을 이해해 질문 언어로 답변을 생성합니다.
- 답변과 함께 **참고한 원문 FAQ의 언어/카테고리**를 제공해 신뢰도를 높였습니다.

## 폴더 구조

```
project/
├── data/
│   └── faq_multilingual.csv        # 4개 언어 FAQ 40건 (샘플 데이터)
├── rag_pipeline.py                 # RAG 핵심 로직 (노트북/앱 공용 모듈)
├── app.py                          # Streamlit 배포용 웹 앱
├── RAG_다국어_CS에이전트_소스코드.ipynb  # 제출용 소스코드 노트북
├── requirements.txt
└── README.md
```

## 로컬 실행 방법

```bash
pip install -r requirements.txt

# Anthropic API 키 등록
export ANTHROPIC_API_KEY="sk-ant-..."      # (Windows: set ANTHROPIC_API_KEY=...)

# 노트북 실행
jupyter notebook RAG_다국어_CS에이전트_소스코드.ipynb

# 또는 웹앱 실행
streamlit run app.py
```

## Streamlit Community Cloud 배포 방법

1. 이 `project` 폴더 전체를 새 GitHub 레포지토리에 업로드합니다.
2. https://share.streamlit.io 접속 → **New app** → 레포/브랜치 선택 → Main file path에 `app.py` 지정.
3. **Settings > Secrets** 에 아래 내용을 등록합니다.
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. **Deploy** 클릭 → 발급된 URL이 과제 제출용 "최종 배포 웹 URL"입니다.

> 최초 배포 시 다국어 임베딩 모델(`sentence-transformers`)을 다운로드하므로 첫 로딩이 다소 걸릴 수 있습니다.

## 과제 제출 체크리스트

- [ ] `RAG_다국어_CS에이전트_소스코드.ipynb` 안에 이름/학번 기입
- [ ] `app.py` 상단 주석에 이름/학번 기입
- [ ] Streamlit Cloud 배포 완료 후 URL을 노트북 "6. 배포" 섹션에 기입
- [ ] 프로젝트 결과 보고서(Word) 별도 작성
- [ ] Slack DM으로 곽경일 강사에게 (배포 URL / .ipynb / 보고서) 제출

## 향후 개선 아이디어

- FAQ를 실제 매뉴얼/약관 PDF로 확장해 문서 단위 RAG로 고도화
- 지원 언어 확대 (베트남어, 스페인어 등)
- 대화 히스토리 기반 멀티턴 상담 지원
