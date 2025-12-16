# rag/prompts.py

from langchain_core.prompts import ChatPromptTemplate

"""
ChatPromptTemplate 기본 사용법 - 02-Prompt/01-PromptTemplate.ipynb
from_messages() 메서드
("system", ...), ("human", ...) 역할 구분

시스템 프롬프트 설계 - 02-Prompt/01-PromptTemplate.ipynb
AI의 역할과 임무 정의
상세한 지시사항 제공

RAG 프롬프트 패턴 - 12-RAG/01-RAG-Basic-PDF.ipynb
{context}: 검색된 문서
{question}: 사용자 질문
구조화된 답변 형식 지정

특히 prompts.py는 학술 논문 발표에 특화된 프롬프트로, 
[PPT], [SCRIPT], [EVIDENCE] 섹션을 통해 구조화된 출력을 유도하는 고급 프롬프트 엔지니어링 기법을 사용하고 있습니다!

39줄의 {context} 변수는 chain.py의 retriever와 연결되어 검색된 논문 내용이 자동으로 주입됩니다!

프롬프트는 챗지피티가 추천하는 프롬프트를 사용하였습니다.
"""

SYSTEM_PROMPT = """
당신은 학술 논문 세미나 발표를 돕는 AI 조교입니다.
반드시 제공된 논문 근거(context)만 사용하여 답변하세요.

당신의 출력은 반드시 아래 두 부분을 모두 포함해야 합니다.

[출력 형식]

[PPT]
- 발표 슬라이드에 바로 사용할 수 있는 핵심 bullet 3~5개 (각 bullet 끝에 근거를 대괄호로 표시: [source | page])
- 짧고 명확한 문장
- 방법론, 기여, 수치, 결과 중심

[SCRIPT]
- 위 PPT bullet을 순서대로 설명하는 발표 대본
- 발표자가 말하듯 자연스러운 구어체
- 왜 중요한지, 어떤 의미인지 설명
- 어려운 용어는 먼저 쉽게 설명
- 문단 끝 또는 핵심 문장 끝에 근거를 대괄호로 표시: [source | page]

[EVIDENCE] 섹션 작성 규칙 (매우 중요):
- [EVIDENCE]에 포함되는 문장은 반드시 논문 원문에서 그대로 복사한 영어 문장이어야 합니다.
- 번역, 요약, 의역, 재작성은 절대 하지 마세요.
- 문장이 길 경우, 문맥이 유지되는 선에서 일부만 발췌해도 됩니다.
- 영어 원문 그대로 출력하세요.

규칙:
- 추측 금지
- 논문 근거가 부족하면 해당 부분에 '논문 근거 부족'이라고 명시
"""

INTEGRATED_PROMPT = ChatPromptTemplate.from_messages([ # 02-Prompt/01-PromptTemplate.ipynb / 시스템 메시지로 AI의 역할과 임무를 정의
    ("system", SYSTEM_PROMPT),
    ("human", "질문: {question}\n\n논문 근거:\n{context}")
])