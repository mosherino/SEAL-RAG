### Cached Entities Update Agent

from typing import List, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.states import State, SimpleTriplet, RepairDecisionOutput
from src.modules.configuration import Configuration



class EntityTripletList(BaseModel):
    """LLM output per document"""
    triplets: List[SimpleTriplet] = Field(default_factory=list)

class EntityState(TypedDict):
    """State structure for the document grader containing all necessary context."""
    
    doc_to_extract: Document
    user_query: str
    micro_query: Optional[str]
    cached_entities: Optional[List[SimpleTriplet]]
    repair_decision: Optional[RepairDecisionOutput]


def _create_graphrag_extraction_prompt() -> ChatPromptTemplate:
    """Targeted GraphRAG-style extraction prompt that focuses on the missing link."""
    
    system_prompt = """<ROLE>
You are a knowledge graph extraction assistant. Identify NODES (entities) and EDGES (relationships) from documents.
</ROLE>

<DEFINITIONS>
NODES = Entities (people, places, organizations, etc.)
EDGES = Relationships connecting nodes (played_for, located_in, founded_by, etc.)
GOAL = Extract triplets: NODE1 --EDGE--> NODE2
</DEFINITIONS>

<ENTITY_TYPES>
Use this small static backbone (do not invent new types; use OTHER if unsure):
- PERSON, ORGANIZATION, LOCATION, EVENT, DATE, WORK, LAW/POLICY, CONCEPT, OTHER
</ENTITY_TYPES>

<RELATION_TYPES>
Canonical examples (non-exhaustive; if no clear canonical mapping, use a short literal relation phrase from the sentence):
- located_in, headquartered_in, founded_by, established_by, member_of, worked_at, employed_by,
  created_by, invented_by, authored_by, part_of, subclass_of, same_as, also_known_as,
  occupation, notable_for, award, nationality, birth_date, birth_place, used_in, defined_as,
  is_a, related_to
</RELATION_TYPES>

<RULES>
1) TARGETED-FIRST: Prefer triplets that directly resolve the missing link implied by REPAIR_REASONING and/or LAST_MICRO_QUERY.
2) EXPLICIT-ONLY: Extract facts explicitly stated in SOURCE_DOCUMENT (no inference/generalization).
3) VERBATIM CONSTRAINT: Subject and object strings must appear verbatim in SOURCE_DOCUMENT.
4) NAMING: Use a canonical relation name when obvious; otherwise a short literal phrase from the sentence.
5) HINT BIAS: Prefer ENTITY_TYPE_HINTS and RELATION_HINTS when they apply; otherwise fall back to the static sets above.
6) SCORING: Assign confidence (0-100) and relevance (0-100). Triplets that address the missing link get the highest relevance.
7) COVERAGE: Output up to 10 triplets; only include items that are explicit and relevant. It is valid to output 0 if nothing relevant exists.
</RULES>

<SCORING>
CONFIDENCE:
- 95-100 direct quote; 85-94 clear statement; 70-84 implication; 55-69 inferred; 40-54 weak; 25-39 speculative; 0-24 uncertain
RELEVANCE:
- 95-100 DIRECT ANSWER; 85-94 KEY SUPPORT; 70-84 IMPORTANT CONTEXT; 55-69 USEFUL DETAIL; 40-54 RELATED; 25-39 TANGENTIAL; 0-24 IRRELEVANT
</SCORING>"""

    human_prompt = """<ORIGINAL_QUESTION>
{user_query}
</ORIGINAL_QUESTION>

<REPAIR_REASONING>
{repair_reasoning}
</REPAIR_REASONING>

<LAST_MICRO_QUERY>
{last_micro_query}
</LAST_MICRO_QUERY>

<ENTITY_TYPE_HINTS>
{entity_type_hints}
</ENTITY_TYPE_HINTS>

<RELATION_HINTS>
{relation_hints}
</RELATION_HINTS>

<SOURCE_DOCUMENT>
{document_text}
</SOURCE_DOCUMENT>

<think>
1) Identify explicit relations that directly resolve the missing link implied by REPAIR_REASONING/LAST_MICRO_QUERY.
2) If none, select explicit key-support relations that are necessary stepping stones.
3) Reject any relation where subject or object does not appear verbatim in SOURCE_DOCUMENT.
4) Rank by relevance to the missing link, then by confidence.
</think>

<EXTRACTION_INSTRUCTIONS>
Output triplets:
- subject, subject_type, relation, object, object_type, source_doc_title, confidence, relevance
</EXTRACTION_INSTRUCTIONS>

<OUTPUT_FORMAT>
Return a list field named 'triplets'. Each item must include:
- subject, subject_type, relation, object, object_type, source_doc_title, confidence, relevance
</OUTPUT_FORMAT>

<VALIDATION>
- Subject and object must appear verbatim in SOURCE_DOCUMENT.
- No inference/generalization.
- Prefer 0 triplets over speculative output.
- Up to 10 triplets; include only those relevant to the missing link or its key support.
</VALIDATION>

<RETURN>
Provide a list field named 'triplets'.
</RETURN>"""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])


# Note: Removed all complex conversion and parallel processing functions - simplified approach


async def cached_entities_update(state: EntityState, config: RunnableConfig) -> State:
    """
    Simple entity extraction using SimpleTriplet structure - single LLM call.
    
    Args:
        state: Current graph state containing documents
        config: Runtime configuration
        
    Returns:
        State: Updated state with readable chains for to_repair
    """
    print("---CACHED ENTITIES UPDATE (SimpleTriplet)---")
    
    # Get documents and user query
    document = state.get("doc_to_extract", None)
    user_query = state["user_query"]
    
    # NEW: dynamic conditioning inputs
    last_micro_query = state.get("micro_query") or ""
    repair_decision = state.get("repair_decision", None)
    repair_reasoning = getattr(repair_decision, "reasoning", "") if repair_decision else ""
    
    try:
        # Get configuration and build LLM
        configurable = config.get("configurable", {})
        configuration = Configuration(**configurable)
        llm = configuration.build_extraction_llm()
        
        # Create structured output LLM for DocumentExtraction
        structured_extractor = llm.with_structured_output(EntityTripletList)
        
        # Create extraction chain
        extraction_prompt = _create_graphrag_extraction_prompt()
        extraction_chain = extraction_prompt | structured_extractor
        
        # Process single document
        if not document:
            print("No document to extract from")
            return {
                "cached_entities": []
            }
            
        # Window document to focus terms derived from question + reasoning + last micro query
        def _extract_focus_terms(text: str, k: int = 18):
            import re
            from collections import Counter
            toks = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text or "")
            toks = [t for t in toks if len(t) > 2]
            return [w for w, _ in Counter(toks).most_common(k)]
        
        focus_terms = _extract_focus_terms(" ".join([user_query or "", repair_reasoning or "", last_micro_query or ""]))
        
        import re as _re
        sentences = _re.split(r"(?<=[.!?])\s+", document.page_content or "")
        keep = []
        seen = set()
        for s in sentences:
            if any(ft.lower() in s.lower() for ft in focus_terms):
                key = s.strip()[:200]
                if key not in seen:
                    seen.add(key)
                    keep.append(s)
                if len(keep) >= 30:
                    break
        
        doc_title = document.metadata.get("title", "Document")
        doc_body = " ".join(keep) if keep else (document.page_content[:2000] or "")
        document_text = f"--- {doc_title} ---\n{doc_body}"
        
        # Derive dynamic type/relation hints
        def _derive_type_hints(text: str):
            t = (text or "").lower()
            hints = set()
            if any(x in t for x in ["company", "inc", "corp", "ltd", "org", "organization"]):
                hints.add("ORGANIZATION")
            if any(x in t for x in ["city", "country", "state", "province", "region", "located"]):
                hints.add("LOCATION")
            if any(x in t for x in ["born", "died", "author", "researcher", "person", "who is"]):
                hints.add("PERSON")
            if any(x in t for x in ["event", "conference", "tournament", "summit"]):
                hints.add("EVENT")
            if any(x in t for x in ["year", "date", "when", "timeline"]):
                hints.add("DATE")
            if any(x in t for x in ["paper", "book", "journal", "article", "publication"]):
                hints.add("WORK")
            if any(x in t for x in ["law", "act", "policy", "regulation", "statute"]):
                hints.add("LAW/POLICY")
            if not hints:
                hints.add("OTHER")
            return ", ".join(sorted(hints))
        
        def _derive_relation_hints(text: str):
            t = (text or "").lower()
            rels = set()
            if any(x in t for x in ["founder", "founded", "established", "co-founded"]):
                rels.add("founded_by")
            if any(x in t for x in ["author", "wrote", "written by"]):
                rels.add("authored_by")
            if "born" in t:
                rels.update(["birth_date", "birth_place"])
            if any(x in t for x in ["headquarter", "hq"]):
                rels.add("headquartered_in")
            if "located" in t:
                rels.add("located_in")
            if any(x in t for x in ["member", "affiliate", "subsidiary", "unit"]):
                rels.update(["member_of", "part_of"])
            if any(x in t for x in ["occupation", "profession", "known for", "notable for"]):
                rels.update(["occupation", "notable_for"])
            if any(x in t for x in ["nationality", "citizen"]):
                rels.add("nationality")
            if any(x in t for x in ["defined as", "is a", "type of", "kind of"]):
                rels.update(["defined_as", "is_a", "subclass_of"])
            return ", ".join(sorted(rels)) if rels else "related_to"
        
        entity_type_hints = _derive_type_hints(" ".join([user_query or "", repair_reasoning or "", last_micro_query or ""]))
        relation_hints = _derive_relation_hints(" ".join([user_query or "", repair_reasoning or "", last_micro_query or ""]))
        
        print("Extracting triplets from document (dynamic focus)...")
        
        # Single LLM call to extract triplets with dynamic conditioning
        extraction_result = await extraction_chain.ainvoke({
            "user_query": user_query,
            "repair_reasoning": repair_reasoning,
            "last_micro_query": last_micro_query,
            "entity_type_hints": entity_type_hints,
            "relation_hints": relation_hints,
            "document_text": document_text
        })
        
        triplets = extraction_result.triplets if extraction_result else []
        
        # Light evidence-based filtering only (no schema-specific normalization)
        filtered = []
        for t in triplets:
            if t.subject and t.object and (t.subject in document_text) and (t.object in document_text):
                filtered.append(t)
        
        return {
            "cached_entities": filtered
        }
        
    except Exception as e:
        print(f"Error in extraction: {e}")
        return {
            "cached_entities": []
        }




