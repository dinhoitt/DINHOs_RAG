# rag/chain.py

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from rag.prompts import INTEGRATED_PROMPT


def format_docs(docs):
    """
    Retriever 결과를 context 문자열로 변환
    (출처 PDF와 페이지 정보 포함)
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        content = d.page_content.strip().replace("\n", " ")
        # 너무 긴 chunk는 잘라서 인용 안정화
        content = content[:1200]
        blocks.append(f"[E{i}] [{source} | page {page}] {content}")

    joined = "\n".join(blocks)
    return joined[:12000]  # 토큰 과다 방지


def build_rag_chain(retriever):
    """
    단일 RAG 체인:
    - 하나의 검색
    - 하나의 프롬프트
    - PPT + SCRIPT 동시 출력
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    chain = (
        {
            "question": lambda x: x["question"],
            "context": (lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        }
        | INTEGRATED_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
