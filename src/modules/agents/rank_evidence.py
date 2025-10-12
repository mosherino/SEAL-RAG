from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

from src.states import State, SimpleTriplet
from src.modules.configuration import Configuration


class RelevanceScoringOutput(BaseModel):
    # Entity-first selection:
    # The model returns 1-based indices (matching <ENTITIES> section in the prompt)
    # for the minimal set of entities needed to answer.
    entities_to_keep: List[int] = []  # 1-based entity indices (<= 10)


def _format_indexed_entities(entities: List[SimpleTriplet], max_n: int = 50) -> str:
    shown = entities[:max_n]
    lines = []
    for i, e in enumerate(shown, 1):
        lines.append(f"[{i}] [{e.subject_type}] {e.subject} --{e.relation}--> [{e.object_type}] {e.object}")
    return "\n".join(lines) if lines else "None"


def _format_indexed_documents(documents: List[Document], max_k: int = 50, snippet_len: int = 6000) -> str:
    shown = documents[:max_k]
    blocks = []
    for i, d in enumerate(shown, 1):
        title = (d.metadata or {}).get("title", "Document")
        snippet = (d.page_content or "")[:snippet_len]
        blocks.append(f"[{i}] {title}\n{snippet}")
    return "\n\n".join(blocks) if blocks else "None"


def _create_rank_prompt() -> ChatPromptTemplate:
    # Prompt guides the model to pick ENTITIES first (minimal sufficient chain).
    # We still show DOCUMENTS for context, but selection is by entity indices only.
    system = """You are a relevance ranker. Be conservative and prefer fewer, higher-quality items."""
    human = """QUESTION:
<QUESTION>
{question}
</QUESTION>

REPAIR REASONING (guides which chain is sufficient):
<REPAIR_REASONING>
{repair_reasoning}
</REPAIR_REASONING>

ENTITIES (indexed):
<ENTITIES>
{indexed_entities}
</ENTITIES>

DOCUMENTS (indexed):
<DOCUMENTS>
{indexed_documents}
</DOCUMENTS>

TASKS:
1) Prioritize only items that support the minimal sufficient chain implied by <REPAIR_REASONING>. Deprioritize unrelated clubs/entities.
2) Select the minimal sufficient subset of ENTITIES (<= 10) as a list of indices that together support the required chain(s) with high confidence. Choose the FEWEST entities needed to answer.

Return JSON with:
- entities_to_keep: [indices]"""
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])


async def rank_evidence(state: State, config: RunnableConfig) -> State:
    print("---RANK EVIDENCE---")
    user_query = state["user_query"]
    documents = state.get("documents", []) or []
    entities = state.get("cached_entities", []) or []

    indexed_entities = _format_indexed_entities(entities, max_n=30)
    indexed_documents = _format_indexed_documents(documents, max_k=50, snippet_len=600)
    repair_reasoning = ""
    repair_decision = state.get("repair_decision")
    if repair_decision and getattr(repair_decision, "reasoning", None):
        repair_reasoning = repair_decision.reasoning

    configurable = config.get("configurable", {})
    configuration = Configuration(**configurable)
    llm = configuration.build_llm()
    structured = llm.with_structured_output(RelevanceScoringOutput)

    prompt = _create_rank_prompt()
    chain = prompt | structured

    result = await chain.ainvoke({
        "question": user_query,
        "repair_reasoning": repair_reasoning,
        "indexed_entities": indexed_entities,
        "indexed_documents": indexed_documents,
    })

    # Determine entities_to_keep; bounds-check 1-based indices.
    # If none are returned, keep the first entity (when available) to ensure progress.
    entities_to_keep = [idx for idx in (result.entities_to_keep or []) if 1 <= idx <= len(entities)]
    if not entities_to_keep and entities:
        entities_to_keep = [1]

    # Build relevance_documents by deriving titles from the selected entities' source documents.
    # This avoids index drift across loops by using stable document titles.
    def _get_title(doc: Document) -> str:
        return (doc.metadata or {}).get("title", "Document")

    # Collect titles from entities_to_keep via each entity's source_doc_title.
    selected_titles = []
    for idx in entities_to_keep:
        if 1 <= idx <= len(entities):
            src_title = getattr(entities[idx - 1], "source_doc_title", None)
            if src_title:
                selected_titles.append(src_title)

    # Filter the current candidate documents to only those whose title is in the selected set.
    relevance_documents = [d for d in documents if _get_title(d) in set(selected_titles)]

    # Compute relevance_entities directly from entities_to_keep (stable even if document order changes)
    relevance_entities = [entities[i - 1] for i in entities_to_keep if 1 <= i <= len(entities)]

    # Ensure at least one document fallback (first document) so downstream
    # answer generation always has some context.
    if not relevance_documents and documents:
        relevance_documents = documents[:1]

    return {
        "relevance_documents": relevance_documents,
        "relevance_entities": relevance_entities,
    }


