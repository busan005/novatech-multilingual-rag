"""
rag_pipeline.py
----------------
글로벌 이커머스 다국어 CS 상담 에이전트 - RAG 파이프라인 핵심 모듈

이름:  [학생 이름을 입력하세요]
학번:  [학번을 입력하세요]
"""

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "faq_multilingual.csv")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "claude-sonnet-4-5-20250929"


def load_documents(csv_path: str = DATA_PATH):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        content = f"[Q] {row['question']}\n[A] {row['answer']}"
        metadata = {
            "id": int(row["id"]),
            "language": row["language"],
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_vectorstore(documents=None, persist_directory: str = PERSIST_DIR):
    if documents is None:
        documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_vectorstore(persist_directory: str = PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return build_vectorstore(persist_directory=persist_directory)


PROMPT_TEMPLATE = """당신은 글로벌 전자제품 브랜드 'NovaTech'의 다국어 고객센터 상담 에이전트입니다.

아래는 검색된 참고 FAQ 문서들입니다. 사용자가 어떤 언어로 질문하더라도,
반드시 사용자가 질문한 언어와 같은 언어로 답변하세요.
참고 문서의 언어와 사용자 질문 언어가 다르더라도 내용을 정확히 번역·요약해서 답변에 반영하세요.
답변 마지막에는 참고한 원문 FAQ의 언어와 카테고리를 간단히 밝혀주세요.

[검색된 참고 FAQ]
{context}

[사용자 질문]
{question}

[답변]"""


def format_docs(docs):
    lines = []
    for d in docs:
        lines.append(
            f"(언어: {d.metadata.get('language')}, 카테고리: {d.metadata.get('category')})\n"
            f"Q: {d.metadata.get('question')}\nA: {d.metadata.get('answer')}"
        )
    return "\n\n---\n\n".join(lines)


def build_rag_chain(vectorstore, k: int = 3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatAnthropic(
        model=LLM_MODEL_NAME,
        temperature=0.2,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer_question(chain, retriever, question: str):
    sources = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, sources
