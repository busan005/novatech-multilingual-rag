"""
app.py
------
글로벌 이커머스 다국어 CS 상담 에이전트 - Streamlit 배포용 앱
"""

import os
import streamlit as st
from rag_pipeline import load_vectorstore, build_rag_chain

st.set_page_config(page_title="NovaTech 다국어 CS 에이전트", page_icon="🌐", layout="centered")

if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

st.title("🌐 NovaTech 다국어 CS 상담 에이전트")
st.caption(
    "RAG(검색 증강 생성) + LangChain + Claude 기반 다국어 고객센터 챗봇입니다. "
    "한국어·영어·일본어·중국어 중 어떤 언어로 질문하셔도, "
    "언어와 무관하게 관련 FAQ를 찾아 질문하신 언어로 답변해 드립니다."
)

with st.sidebar:
    st.header("ℹ️ 프로젝트 소개")
    st.markdown(
        """
- **핵심 기술**: LangChain, RAG (Chroma VectorDB, 다국어 Embedding), Claude
- **데이터**: NovaTech 브랜드의 반품/AS/배송/제품사용/계정 관련 다국어(4개 언어) FAQ 40건
- **차별점**: 단순 번역이 아니라, 질문 언어와 무관하게 **의미 기반 검색**
  후 질문 언어로 자연스럽게 답변하고 참고한 원문 출처를 함께 제공합니다.
        """
    )
    st.divider()
    example_lang = st.selectbox("예시 질문 언어", ["한국어", "English", "日本語", "中文"])
    examples = {
        "한국어": "해외에서 산 이어폰인데 한국 AS센터에서 고칠 수 있나요?",
        "English": "Can I get a refund if I bought the product overseas?",
        "日本語": "空気清浄機のフィルターはいつ交換すればいいですか?",
        "中文": "国际配送的关税谁来承担?",
    }
    st.code(examples[example_lang], language=None)

if "chain" not in st.session_state:
    with st.spinner("다국어 벡터스토어를 준비하는 중입니다... (최초 1회, 다소 시간이 걸릴 수 있습니다)"):
        vectorstore = load_vectorstore()
        chain, retriever = build_rag_chain(vectorstore)
        st.session_state.chain = chain
        st.session_state.retriever = retriever

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("어떤 언어로든 질문해 보세요 (한국어/영어/일본어/중국어)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                answer = "⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다. Streamlit Cloud의 Secrets에 API 키를 등록해 주세요."
                sources = []
            else:
                answer = st.session_state.chain.invoke(user_input)
                sources = st.session_state.retriever.invoke(user_input)
            st.markdown(answer)

            if sources:
                with st.expander("🔎 참고한 원문 FAQ 보기"):
                    for s in sources:
                        st.markdown(
                            f"**[{s.metadata.get('language')} / {s.metadata.get('category')}]**\n\n"
                            f"- Q: {s.metadata.get('question')}\n"
                            f"- A: {s.metadata.get('answer')}"
                        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
