### To Repair Agent

from typing import List
from src.states import State, SimpleTriplet, RepairDecisionOutput
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from src.modules.configuration import Configuration




def _format_cached_entities_for_repair(cached_entities: List[SimpleTriplet]) -> str:
    """
    Format cached entities for the repair decision prompt.
    
    Args:
        cached_entities: List of cached entity triples
        
    Returns:
        Formatted string for prompt inclusion
    """
    if not cached_entities:
        return "CACHED ENTITIES: None available"
    
    # Limit to prevent token overflow - show up to 100 entities for decision-making
    entities_to_show = cached_entities[:100]
    
    formatted = "CACHED ENTITIES (Knowledge from previous iterations):\n"
    for i, e in enumerate(entities_to_show, 1):
        formatted += (
            f"[{i}] [{e.subject_type}] {e.subject} → {e.relation} → [{e.object_type}] {e.object}\n\n"
        )
    
    if len(cached_entities) > 100:
        formatted += f"... and {len(cached_entities) - 100} more entities\n"
    
    return formatted


def _format_documents_for_repair(documents: List[Document], max_docs: int = 20, snippet_len: int = 1000) -> str:
    """
    Format top document snippets for inclusion in the repair decision prompt to help resolve aliases.
    """
    if not documents:
        return "No documents available."
    out = []
    for d in documents[:max_docs]:
        title = (d.metadata or {}).get("title", "Document")
        snippet = (d.page_content or "")[:snippet_len]
        out.append(f"--- {title} ---\n{snippet}")
    return "\n\n".join(out)

# Association relations used to derive candidate groups/entities
_ASSOCIATION_RELATIONS = {
    "associated_with",
    "improves_relations_between",
    "aims_to_improve_relations_between",
    "produced_to_improve_relations_between",
}

def _extract_candidates_for_repair(cached_entities: List[SimpleTriplet]) -> str:
    """
    Build a parsed candidate list from association relations by splitting multi-group
    objects on generic delimiters and deduplicating.
    """
    if not cached_entities:
        return "None"
    # Collect raw candidate strings from association edges only
    raw: list[str] = []
    for e in cached_entities:
        rel_norm = (e.relation or "").strip().lower().replace(" ", "_")
        if rel_norm in _ASSOCIATION_RELATIONS:
            if e.object:
                raw.append(e.object)
    if not raw:
        return "None"
    # Split on common multi-entity delimiters
    import re
    split_candidates: list[str] = []
    pattern = re.compile(r"\s*(?:,|/|&|\band\b)\s*", re.IGNORECASE)
    for item in raw:
        parts = [p.strip(" \t\n\r,/&") for p in pattern.split(item) if p.strip()]
        split_candidates.extend(parts if parts else [item])
    # Deduplicate while preserving order
    seen = set()
    ordered: list[str] = []
    for c in split_candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(c)
    if not ordered:
        return "None"
    # Format as indexed lines
    lines = [f"[{i}] {name}" for i, name in enumerate(ordered, 1)]
    return "\n".join(lines)

def _create_repair_decision_prompt() -> ChatPromptTemplate:
    """Create template-based logical chain verification prompt with explicit step-by-step entity checking to prevent hallucination."""
    
    system_prompt = """You are a logical chain checker. Use the template exactly.

RULE: Check if connections exist, don't create them."""

    human_prompt = """QUERY:
<QUERY>
{original_query}
</QUERY>

ENTITIES:
<ENTITIES>
{cached_entities_context}
</ENTITIES>

FORMAT ENTITIES CLEARLY:
- Insert a blank line between each entity.
- Keep each entity on a single line: [index] [TYPE] Subject → relation → [TYPE] Object.

DOCUMENTS (snippets to use only as explicit evidence; do not invent aliases):
<DOCUMENTS>
{documents_context}
</DOCUMENTS>

CANDIDATES (parsed from association relations; deduplicated):
<CANDIDATES>
{candidates_context}
</CANDIDATES>

DOCUMENT PRIORITY (for attribute values):
- Use <DOCUMENTS> as the primary source of attribute values (numbers, percents, dates, locations). <ENTITIES> provide candidate/edge scaffolding. If an explicit value appears in <DOCUMENTS>, you MUST use it even if it is not listed in <ENTITIES>.

<BRIDGING_RULES>
- Optional, strictly evidence-backed bridge (generalized; do not overfit):
  Pattern:
    • If entity X is explicitly related to base artifact A via a REPRESENTS relation:
      REPRESENTS ∈ [defined_as, reproduction_of, facsimile_of, depiction_of, copy_of, transcription_of, transliteration_of]
    • AND a DOCUMENT explicitly states that A has an attribute T with value R:
      T ∈ [script, writing_system, language, symbol_system] (non-exhaustive)
      Examples of explicit phrasing (not exhaustive): “A uses R”, “A written in R”, “A in R”.
  Then you MAY derive one bridge for each applicable T:
    • If T ∈ [script, writing_system]:    X → uses_script → R
    • If T = language:                    X → uses_language → R
    • If T = symbol_system:               X → uses_symbol_system → R
  Constraints:
    • Evidence for both parts must be explicit and verbatim in <DOCUMENTS>/<ENTITIES>.
    • At most one derived bridge per T; no other bridges.
    • Tag bridge=True in your internal reasoning; set confidence ≤ 75 and relevance ≤ 70 if you materialize it.
    • If any condition is missing, do NOT derive.
</BRIDGING_RULES>

SCOPE RULES (apply strictly):
- If the query implies "a", "an", or "any" (e.g., "a club"), ONE complete chain is sufficient. Do NOT require more; ignore missing links for other entities.
- If the query implies "all" or "every", require complete chains for ALL relevant starting entities; if any required chain is missing, return No.
- If the query implies "the" (a specific singular entity), require a UNIQUE, unambiguous chain; if multiple candidates remain or none is unique, return No.
- Treat entity variants as equivalent ONLY if explicitly supported by DOCUMENTS (e.g., "also known as", "formerly", or clear co-mention). Do NOT invent links.
 - ATTRIBUTE INTENT: If the question seeks a value-like attribute (percentage/%/rate/number/count/total, date/year/when, location/where/located/headquarters), mark intent=True; else False.
 - If intent=True and multiple associated candidates exist, evaluate each candidate separately and select ANY candidate with an explicit attribute chain; do NOT require all candidates.
 - If intent=True, a complete chain MUST include an explicit attribute value (e.g., Candidate → percentage_of_global_population → 17%). If no explicit value exists for any candidate, return No.

DISTRACTOR FILTER:
- Ignore unrelated or distractor entities and edges that are not necessary to satisfy the scope. Do NOT penalize missing links that are irrelevant to the minimal sufficient chain.

SYSTEMATIC REASONING TEMPLATE:

1. QUERY DECOMPOSITION:
   - Core question: [What specific information is being requested?]
   - Query completeness: [Is the question well-formed and answerable?]
   - Scope requirement: [a/an/any vs all/every vs the (specific)]
   - Success criteria: [What constitutes a complete answer under the scope?]

2. PATH DISCOVERY:
   - Starting entities: [Identify ALL plausible starting points relevant to the scope]
   - Candidate parsing (multi-group):
     • If an association object lists multiple candidates joined by delimiters (e.g., "and", "&", "/", ","), SPLIT into separate candidates (trim whitespace/punctuation).
     • Treat each candidate as independent for evaluation; do not assume joint properties across the grouped string.
   - Use candidates from <CANDIDATES> only; do not infer new candidates beyond this set.
   - Candidate evaluation (association pivot): derive candidates from any association relation (associated_with/improves_relations_between/aims_to_improve_relations_between/produced_to_improve_relations_between). For each candidate independently, attempt Subject → association → Candidate → has_attribute/percentage/located_in/etc. → Value (explicit).
   - DOCUMENT SEARCH PROCEDURE (when ATTRIBUTE INTENT=True):
     • For each candidate, scan <DOCUMENTS> first for explicit statements of the target attribute. If a concrete value token is present (e.g., a number, percent, date, or place), record [candidate, value, doc title]. If none found in <DOCUMENTS>, then check <ENTITIES> for an explicit value.
   - Candidate ranking (when ATTRIBUTE INTENT=True):
     • Prefer candidates that have an explicit attribute value present in ENTITIES or DOCUMENTS.
     • Break ties by shortest, most explicit chain (fewer hops, direct statements).
     • Select the highest-ranked candidate that completes the chain.
   - Possible chains: [For each starting entity, list reasoning paths]
   - Chain completeness: [Which paths lead from question to answer?]

3. RELATIONSHIP ANALYSIS:
   - Direct matches: [Exact entity and relationship matches]
   - Evidence-backed equivalence: [Only if DOCUMENTS explicitly support alias/equivalence]
   - Derived bridges (apply <BRIDGING_RULES> strictly): [List any X → uses_* → R with bridge=True; confidence ≤ 75, relevance ≤ 70]
   - Missing links: [Only for entities required under the scope]

4. SUFFICIENCY ASSESSMENT (SCOPE/INTENT-AWARE):
   - If (scope = a/an/any OR the target is underspecified among multiple associated candidates) AND intent=True: ONE complete candidate chain with an explicit attribute value suffices; otherwise No.
     • Return Yes as soon as ANY candidate yields an explicit attribute value chain.
     • Name the selected candidate and the value in the justification.
   - If intent=False: apply standard scope rules without the attribute-value requirement.
   - If scope = all/every: Are ALL required chains complete? If any required chain is missing, return No.
   - If scope = the (specific): Is there exactly ONE unambiguous complete chain?

5. DECISION SYNTHESIS (CONCISE):
   - Answer feasibility (scope-aware): [Yes/No]
   - Minimal justification: [1–2 short sentences focusing only on the confirmed chain or the single missing link]
   - If Yes and ATTRIBUTE INTENT=True, include an evidence trace: Selected Candidate = [...]; Value = [...]; Source = [Doc title or ENTITIES]

EXAMPLE DEMONSTRATIONS:

COMPLETE PATH EXAMPLE:
Query: "Where was the university that educated the Google founder located?"
Entities: Larry Page → educated_at → Stanford University, Stanford University → located_in → California

1. QUERY DECOMPOSITION:
   - Core question: Location of university where Google founder was educated
   - Query completeness: Well-formed, answerable
   - Scope requirement: Specific location for specific person's university
   - Success criteria: Complete chain from founder → university → location

2. PATH DISCOVERY:
   - Starting entities: Larry Page (identified as Google founder)
   - Possible chains: Larry Page → educated_at → Stanford University → located_in → California
   - Chain completeness: Full path exists from person to final location

3. RELATIONSHIP ANALYSIS:
   - Direct matches: Exact entities and relationships found
   - Semantic matches: Larry Page clearly identified as Google founder
   - Missing links: None

4. SUFFICIENCY ASSESSMENT:
   - Available paths: One complete chain
   - Path confidence: High (direct, exact matches)
   - Query satisfaction: Fully meets requirements

DECISION: Yes

INCOMPLETE QUERY EXAMPLE:
Query: "When a Man Falls in Love stars Song Seung-heon and which South Korean actor, born on ?"

1. QUERY DECOMPOSITION:
   - Core question: Birth date information, but incomplete
   - Query completeness: Malformed - "born on ?" lacks the actual date/information requested
   - Cannot proceed with path analysis due to incomplete query structure

DECISION: No (Query malformed)

INSUFFICIENT EVIDENCE EXAMPLE:
Query: "Where was the CEO of Amazon educated?"
Entities: Jeff Bezos → founded → Amazon, Princeton University → located_in → New Jersey

1. QUERY DECOMPOSITION:
   - Core question: Educational institution of Amazon's CEO
   - Query completeness: Well-formed, answerable
   - Scope requirement: Education location for specific person
   - Success criteria: Chain from CEO → educational institution

2. PATH DISCOVERY:
   - Starting entities: Jeff Bezos (identified through Amazon connection)
   - Possible chains: Jeff Bezos → educated_at → [Missing]
   - Chain completeness: Incomplete - education link missing

3. RELATIONSHIP ANALYSIS:
   - Direct matches: Jeff Bezos → founded → Amazon exists
   - Missing links: Jeff Bezos → educated_at → University

4. SUFFICIENCY ASSESSMENT:
   - Available paths: No complete chains to education
   - Path confidence: N/A (critical link missing)
   - Query satisfaction: Cannot be met with available evidence

DECISION: No

DECISION: [Yes/No] based on query validity AND found connections."""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])


async def to_repair(state: State, config: RunnableConfig) -> State:
    """
    Evaluate whether we can answer the original query based on available evidence.
    
    Uses cached entities and document evidence to determine if we have sufficient
    information to provide a complete answer or need to continue the repair process.
    
    Args:
        state: Current graph state containing user_query, cached entities, and documents
        config: Runtime configuration
        
    Returns:
        State: Updated state with repair decision
    """
    print("---TO REPAIR?---")
    
    # Get configuration
    configurable = config.get("configurable", {})
    configuration = Configuration(**configurable)

    # Simple fallback: state → configuration → default
    loop_limit = configuration.repair_loop_limit 
    
    # Get inputs from state
    user_query = state["user_query"]
    cached_entities = state.get("cached_entities", [])
    documents = state.get("documents", [])

    
    print(f"Evaluating repair decision with {len(cached_entities)} cached entities and {len(documents)} documents")
    
    # Build LLM with structured output
    llm = configuration.build_llm()
    structured_llm = llm.with_structured_output(RepairDecisionOutput)
    
    # Create decision chain
    decision_prompt = _create_repair_decision_prompt()
    decision_chain = decision_prompt | structured_llm
    
    # Format evidence contexts - include entities, brief document snippets, and parsed candidates
    cached_entities_context = _format_cached_entities_for_repair(cached_entities)
    documents_context = _format_documents_for_repair(documents)
    candidates_context = _extract_candidates_for_repair(cached_entities)
    # document_evidence_context removed (not used)
    
    try:
        print("Making repair decision with LLM...")
        
        # Get repair decision from LLM with entities and document snippets
        decision_result = await decision_chain.ainvoke({
            "original_query": user_query,
            "cached_entities_context": cached_entities_context,
            "documents_context": documents_context,
            "candidates_context": candidates_context
        })
        
        print(f"Decision: {decision_result.can_answer_original_query}")
        # Make reasoning concise for logs
        concise_reason = decision_result.reasoning
        if concise_reason and len(concise_reason) > 400:
            concise_reason = concise_reason[:400] + "..."
        print(f"Reasoning: {concise_reason}")
        
        return {
            "user_query": user_query,
            "repair_decision": decision_result,  # Store the entire Pydantic object
            "repair_loop_limit": loop_limit
        }
        
    except Exception as e:
        print(f"Error in repair decision: {e}")
        # Default to continuing repair process on error
        error_decision = RepairDecisionOutput(
            can_answer_original_query="no",
            reasoning=f"Error occurred during repair decision: {str(e)}"
        )
        
        return {
            "user_query": user_query,
            "repair_decision": error_decision,
            "repair_loop_limit": loop_limit
        }


def route_after_repair_decision(state: State):
    """
    Route based on repair decision.
    
    Args:
        state: Current state with repair_decision field
        
    Returns:
        str: Next node to execute
    """
    print("---ROUTING AFTER REPAIR DECISION---")
    
    repair_decision = state.get("repair_decision")
    
    if repair_decision and repair_decision.can_answer_original_query.lower() == "yes":
        print("✅ Ready to answer - proceeding to rank_evidence")
        return "rank_evidence"
    
    # Not ready: check loop count
    loop_count = state.get("repair_loop_count") or 0
    loop_limit = state.get("repair_loop_limit") 
    print(f"Loop count: {loop_count}, Loop limit: {loop_limit}")
    if loop_count >= loop_limit:
        print("🔁 Loop limit reached, proceeding to rank_evidence")
        return "rank_evidence"
    
    print("❌ Not ready to answer - proceeding to micro_query_agent")
    return "micro_query_agent"
