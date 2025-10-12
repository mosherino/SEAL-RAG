### Generate

from langchain_core.runnables import RunnableConfig
from src.states import State
from src.modules.configuration import Configuration
from langchain_core.prompts import ChatPromptTemplate


def _create_answer_generation_prompt() -> ChatPromptTemplate:
    """Create prompt for final answer generation using cached entities and documents."""
    
    # Enhanced system prompt for multi-hop reasoning
    system = """\
You are an expert answer synthesizer for a multi-hop question answering system. Your job is to provide SHORT, ACCURATE answers based ONLY on the provided entities and documents.

You have access to:
1. **CURATED ENTITIES**: Key facts extracted and filtered for relevance across multiple reasoning steps.
2. **SUPPORTING DOCUMENTS**: Additional context and verification.
3. **ORIGINAL QUESTION**: The complex question requiring multi-step reasoning.
4. **REPAIR REASONING**: The minimal sufficient chain guidance from the decision step.

CRITICAL RULES:
1. **BE PRECISE**: Answer in 2-4 words maximum.
2. **BE CONCISE**: Only provide the essential information needed to directly answer the question.
3. **BE STRICT**: Only use facts explicitly stated in the entities and documents.
4. **BE HONEST**: If you cannot fully answer based on the given data, return "I don't know."
5. **OMIT UNNECESSARY DETAILS**: Do NOT include extra facts, side notes, or additional context, even if they are correct, unless they are strictly required to answer the question correctly.
6. **GIVE THE BEST ANSWER**: Give the best specific answer possible. If not possible, give the best answer you can at the scope.

SYNTHESIS STRATEGY:
- **Start with entities**: These represent the most important, curated facts.
- **Use documents for context**: Only when the entities alone do not fully resolve the question.
- **Connect the dots**: Only include facts necessary to complete the reasoning chain. Avoid adding extra facts that are not needed for the specific answer.
- **Multi-hop reasoning**: Connect the minimum number of steps required to answer the question correctly.
- **Missing links**: If any essential link is missing, return "I don't know."
- **Focus on minimality**: Extra details that do not contribute to answering the question should be excluded.

OUTPUT REQUIREMENTS:
- **Length**: 2-4 words maximum.
- **Format**: Direct answer only. Do not provide explanations unless the reasoning is inherently multi-hop and all steps are required.
- **Unknown answers**: If insufficient evidence is provided, respond with exactly: "I don't know."

EXAMPLES:
- Good: "Paris."
- Good: "Yes."
- Good: "I don't know."
- Bad: "The capital of France is Paris, which is a popular tourist destination." → Extra information not required.
- Bad: "Based on the provided evidence, it seems that..." → Too verbose.
- Bad: "The answer is Paris because..." → Explanatory phrases are unnecessary.

The goal is to deliver the **most minimal, factually complete answer** that directly resolves the question.
"""

    human = """\
**QUESTION:**
{question}

**REPAIR REASONING:**
{repair_reasoning}

**CURATED ENTITIES:**
{cached_entities}

**DOCUMENTS:**
{relevant_documents}

Provide a precise 2-4 words final answer using only the above information."""

    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human)
    ])


def _format_documents(docs):
    """Format documents for prompt display with clear indexing."""
    if not docs:
        return "No documents available."
    
    # Handle both Document objects (with page_content) and GradeDocuments objects (with content)
    formatted_docs = []
    for i, doc in enumerate(docs):
        if hasattr(doc, 'page_content'):
            # Original Document object
            content = doc.page_content
        elif hasattr(doc, 'content'):
            # GradeDocuments object
            content = doc.content
        else:
            # Fallback
            content = str(doc)
        formatted_docs.append(f"[Doc {i+1}]: {content}")
    return "\n\n".join(formatted_docs)


async def generate_agent(state: State, config: RunnableConfig) -> State:
    """
    Generate answer using curated entities and filtered documents from multi-hop reasoning.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    user_query = state["user_query"]
    documents = state.get("documents", [])
    relevance_documents = state.get("relevance_documents", [])
    cached_entities = state.get("cached_entities", [])
    relevance_entities = state.get("relevance_entities", [])
    repair_decision = state.get("repair_decision")
    repair_reasoning = getattr(repair_decision, "reasoning", "") if repair_decision else ""

    # Choose source for generation context by priority:
    # 1) relevance_documents (selected by entities' source titles)
    # 2) documents (raw retrieved)
    documents_for_generation = relevance_documents if relevance_documents else documents

    # Create answer generation prompt
    prompt = _create_answer_generation_prompt()

    # Get configuration and build LLM
    configurable = config.get("configurable", {})
    configuration = Configuration(**configurable)
    llm = configuration.build_llm()

    # Format entities with emphasis on their importance
    entities_str = ""
    chosen_entities = relevance_entities if relevance_entities else cached_entities
    if chosen_entities:
        formatted_entities = []
        for e in chosen_entities:
            # Format: [SUBJECT_TYPE] Subject --relation--> [OBJECT_TYPE] Object (from: source_doc, confidence: X%)
            entity_str = f"[{e.subject_type}] {e.subject} --{e.relation}--> [{e.object_type}] {e.object} \n"
            #entity_str += f" (from: {e.source_doc_title}, confidence: {e.confidence}%)"
            formatted_entities.append(f"• {entity_str}")
        entities_str = "\n".join(formatted_entities)
    else:
        entities_str = "No curated entities available - relying solely on documents."

    # Format documents only if available
    if documents_for_generation:
        formatted_documents = _format_documents(documents_for_generation)
    else:
        formatted_documents = "No documents available."

    # Chain
    rag_chain = prompt | llm

    # RAG generation
    generation = await rag_chain.ainvoke({
        "question": user_query,
        "repair_reasoning": repair_reasoning,
        "cached_entities": entities_str,
        "relevant_documents": formatted_documents
    })

    return {"final_answer": generation.content}