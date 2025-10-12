from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

from src.states import State, SimpleTriplet
from src.modules.configuration import Configuration


class MicroQueryOutput(BaseModel):
    micro_query: str = Field(
        ...,
        description="A short, precise, keyword-focused query that targets the single most critical missing fact needed to answer the original question."
    )


def _format_entities(entities: List[SimpleTriplet], max_n: int = 100) -> str:
    if not entities:
        return "None"
    shown = entities[:max_n]
    lines = []
    for i, e in enumerate(shown, 1):
        lines.append(f"[{i}] [{e.subject_type}] {e.subject} → {e.relation} → [{e.object_type}] {e.object}")
    return "\n\n".join(lines)


def _format_documents(docs: List[Document], max_k: int = 10, snippet_len: int = 1200) -> str:
    if not docs:
        return "No documents available."
    blocks = []
    for i, d in enumerate(docs[:max_k], 1):
        title = (d.metadata or {}).get("title", "Document")
        snippet = (d.page_content or "")[:snippet_len]
        blocks.append(f"--- {title} ---\n{snippet}")
    return "\n\n".join(blocks)


def _create_micro_query_prompt() -> ChatPromptTemplate:
    system = """You are an expert retrieval assistant for a multi-hop question answering system. Your task is to formulate one optimized search query (a micro-query) that will retrieve the missing fact or connection needed to answer the Original Question. Focus on the information gap identified in the Repair Reasoning section and plan a query to find that missing piece. The output must be a single query string, without any additional commentary or reasoning."""

    human = """Provided Context:

Original Question: <ORIGINAL_QUESTION>
{original_question}
</ORIGINAL_QUESTION>

Repair Reasoning: <REPAIR_REASONING>
{repair_reasoning}
</REPAIR_REASONING>

Entities: <ENTITIES>
{entities_block}
</ENTITIES>

Documents: <DOCUMENTS>
{documents_block}
</DOCUMENTS>

Micro Query History: <MICRO_QUERY_HISTORY>
{history_block}
</MICRO_QUERY_HISTORY>

Blocked Titles: <BLOCKED_TITLES>
{blocked_titles}
</BLOCKED_TITLES>

Blocked Keywords: <BLOCKED_KEYWORDS>
{blocked_keywords}
</BLOCKED_KEYWORDS>

Stuck Signal: <STUCK>
{stuck}
</STUCK>

Retrieval Method: <RETRIEVAL_METHOD>
This query will be used as input to a semantic vector-store retriever. Craft a concise set of keywords or phrases (4–12 tokens) that best capture the missing link.
</RETRIEVAL_METHOD>

Iteration: <ITERATION>
{iteration_index}
</ITERATION>

Query Generation Guidelines:

• Target the Missing Information: Base your query on the Repair Reasoning – it describes the critical fact or link that is currently missing. Include relevant Entities or terms that will pinpoint this gap without introducing unrelated information. The query should directly aim to retrieve the fact needed to bridge to the answer.

• Emergency Pivot (for late iterations): If the current Iteration is ≥ 4 or if previous queries have high vocabulary overlap (over ~70% repetition), pivot your search strategy. This means drastically change approach to find new information. For example, consider:
  • Searching by a broader category or class instead of a specific detail.
  • Using a temporal angle (e.g. a year or time period) if time context is relevant.
  • Performing a reverse lookup (e.g. use a known detail from Documents to find related info).
  • Introducing a bridging entity or alternate keyword related to the missing info.
  • Switching the perspective or source-type (e.g. if all prior queries were about one person, query about an event or place linked to them). The goal is to find new leads by changing tactics.

• Avoid Query Redundancy: Do not reuse exact wording from the Micro Query History. Steer clear of repeating any significant unigrams or bigrams from past queries. Use fresh vocabulary or synonyms to cover new ground. (At Iteration ≥ 6, ensure ≥60% of terms are new compared to all previous queries.)

• Drop Anchors: Do NOT include any terms listed in <BLOCKED_KEYWORDS>. Avoid repeating dominant subject/team names and high-frequency tokens from current documents and entities.

• If <STUCK>yes</STUCK>: enforce a strong pivot (different facet such as when/where/why; alternate descriptor/synonym; reverse lookup). Maintain 4–12 content tokens; no stopwords or boolean operators.

• Keyword-Style Query: Formulate the search query in a concise, keyword-focused style. Do not write a full sentence or question; instead, use key terms and names. Aim for 4–12 keyword tokens. Brevity is preferred but not at the expense of clarity.

• Precision & Relevance: Every term must directly target the missing fact.

• Single-Output Only: Return only the newline-free query string — nothing else.

<think>
1) Diagnose why prior queries failed: [redundant vocabulary; wrong facet (who/where/when/why); too specific; too broad; alias mismatch].
2) Choose ONE pivot strategy (different facet; broader/narrower granularity; temporal term; reverse lookup; synonym/alias).
3) Select 4–12 focused content tokens for a semantic vector retriever (no stopwords or boolean operators). Output only the query string.
</think>

Output:
Produce one optimized micro-query (a single line of text) following the above guidelines. This query should be precisely targeted to find the needed information and introduce enough novelty to avoid repetition.

Remember: Output only the query itself — nothing more."""
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])


async def micro_query(state: State, config: RunnableConfig) -> State:
    print("---MICRO QUERY---")
    user_query = state["user_query"]

    # Inputs
    cached_entities = state.get("cached_entities", []) or []
    relevance_entities = state.get("relevance_entities", []) or []
    entities_for_prompt = relevance_entities if relevance_entities else cached_entities

    documents = state.get("documents", []) or []
    relevance_documents = state.get("relevance_documents", []) or []
    documents_for_prompt = relevance_documents if relevance_documents else documents

    repair_decision = state.get("repair_decision")
    repair_reasoning = getattr(repair_decision, "reasoning", "") if repair_decision else ""

    history = state.get("micro_query_history", []) or []
    loop_count = state.get("repair_loop_count") or 0

    # Format blocks
    entities_block = _format_entities(entities_for_prompt, max_n=10000)
    documents_block = _format_documents(documents_for_prompt, max_k=10, snippet_len=12000)
    history_block = "\n".join([f"- {q}" for q in history]) if history else "None"
    # Build blocked titles from current documents to steer exploration
    def _get_title(d: Document) -> str:
        md = d.metadata or {}
        return (md.get("title") or md.get("Title") or md.get("name") or md.get("source") or "Document").strip()
    blocked_titles_list = [_get_title(d) for d in documents_for_prompt]
    blocked_titles = "\n".join([t for t in blocked_titles_list if t]) or "None"

    # NEW: Build blocked keywords from current docs and entities; compute stuck signal
    import re, collections
    def _top_tokens(text: str, k: int = 60):
        toks = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text or "")
        toks = [t.lower() for t in toks if len(t) > 2]
        stop = {"the","and","for","with","that","from","this","they","were","team","season","play","conference"}
        toks = [t for t in toks if t not in stop]
        return [w for w,_ in collections.Counter(toks).most_common(k)]

    doc_text = " ".join([(d.page_content or "")[:2000] for d in documents_for_prompt])
    entity_text = " ".join([e.subject for e in entities_for_prompt] + [e.object for e in entities_for_prompt])
    blocked_keywords_list = sorted(set(_top_tokens(doc_text, 60) + _top_tokens(entity_text, 40)))
    blocked_keywords = ", ".join(blocked_keywords_list) or "None"

    def _tokset(s: str):
        return set((s or "").lower().split())
    # Compute overlap against the most recent history entry (if any)
    if history and len(history) >= 1:
        last_query = history[-1]
        prior_history = history[:-1]
        overlaps = [len(_tokset(last_query) & _tokset(h)) / max(1, len(_tokset(last_query) | _tokset(h))) for h in prior_history]
        overlap_max = max(overlaps) if overlaps else 0.0
    else:
        overlap_max = 0.0
    stuck = "yes" if (overlap_max >= 0.7 or (loop_count or 0) >= 4) else "no"

    # LLM
    configurable = config.get("configurable", {})
    configuration = Configuration(**configurable)
    llm = configuration.build_llm()
    structured = llm.with_structured_output(MicroQueryOutput)

    prompt = _create_micro_query_prompt()
    chain = prompt | structured

    result = await chain.ainvoke({
        "original_question": user_query,
        "repair_reasoning": repair_reasoning,
        "entities_block": entities_block,
        "documents_block": documents_block,
        "history_block": history_block,
        "blocked_titles": blocked_titles,
        "blocked_keywords": blocked_keywords,
        "iteration_index": (loop_count or 0) + 1,
        "stuck": stuck,
    })

    new_query = result.micro_query.strip()
    new_history = history + [new_query] if new_query else history
    new_loop_count = loop_count + 1

    return {
        "micro_query": new_query,
        "micro_query_history": new_history,
        "repair_loop_count": new_loop_count
    }


