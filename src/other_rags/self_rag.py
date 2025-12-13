### Retrieval Grader


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

   


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"

vector_store = PineconeVectorStore(index_name="seal-2wiki-v1", embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vector_store.as_retriever(search_kwargs={"k": 5})#search_kwargs={"k": 3}


### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Custom RAG prompt - similar to rlm/rag-prompt but with stronger grounding
prompt = ChatPromptTemplate.from_messages([
    ("human", """You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question.

If you don't know the answer, just say that you don't know.

Keep the answer concise: use 3-5 words for simple facts, or maximum 1 short sentences for complex answers.

<GROUNDING_RULE>
Base your answer ONLY on the retrieved context below. Do not use any information from your training data or external knowledge.
</GROUNDING_RULE>

Question: {question}

<CONTEXT>
{context}
</CONTEXT>

<INSTRUCTIONS>
Answer based solely on the information within the CONTEXT tags above:
</INSTRUCTIONS>

Answer:""")
])

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()



### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


### Question Re-writer

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        last_retrieved_documents: list of last retrieved documents before filtering
        transform_count: count of query transformations
        generation_attempts: count of generation attempts
    """

    question: str
    generation: str
    documents: List[str]
    last_retrieved_documents: List[str]  # Store last retrieved documents before filtering
    transform_count: int = 0  # Track number of query transformations
    generation_attempts: int = 0  # Track number of generation attempts



### Nodes


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "last_retrieved_documents": documents}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Track generation attempts
    generation_attempts = state.get("generation_attempts", 0) + 1
    print(f"---GENERATION ATTEMPT: {generation_attempts}---")
    
    # If this is a final generation attempt (max reached) and documents are empty,
    # use the last retrieved documents instead
    if generation_attempts >= 2 and not documents:
        last_retrieved = state.get("last_retrieved_documents", [])
        print(f"---USING LAST RETRIEVED DOCUMENTS ({len(last_retrieved)} docs) FOR FINAL ATTEMPT---")
        documents = last_retrieved

    
    # LLM
    llm_generator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt
    system = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use the sentences to answer the question and keep the final answer short and concise (maximum 2-4 words in the final answer). 

    <GROUNDING_RULE>
    Base your answer ONLY on the retrieved context below. Do not use any information from your training data or external knowledge.
    </GROUNDING_RULE>
    """
    # Extract text from documents
    relevance_context = format_docs(documents)
    last_retrieved = state.get("last_retrieved_documents", [])
    last_retrieved_context = format_docs(last_retrieved)
    
    # Prompt with labeled sections
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question:\n\n{question}\n\n"
                "Relevance Context (Graded as Relevant):\n{relevance_context}\n\n"
                "Last Retrieved Context (All Candidates):\n{last_retrieved_context}\n\n"
                "Answer the question (maximum 2-4 words)."
            ),
        ]
    )

    generate_chain = re_write_prompt | llm_generator | StrOutputParser()

    # Invoke with extracted contexts
    generation = generate_chain.invoke({
        "question": question,
        "relevance_context": relevance_context,
        "last_retrieved_context": last_retrieved_context,
    })
    
    return {
        "documents": documents, 
        "question": question, 
        "generation": generation,
        "generation_attempts": generation_attempts
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Get current transform count, defaulting to 0 if not present
    transform_count = state.get("transform_count", 0)
    
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    
    # Increment transform count
    transform_count += 1
    print(f"---TRANSFORM COUNT: {transform_count}---")
    
    return {
        "documents": documents, 
        "question": better_question, 
        "transform_count": transform_count
    }


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    # Get transform count, defaulting to 0 if not present
    transform_count = state.get("transform_count", 0)
    
    # Check if we've reached the maximum number of transformations (2)
    if transform_count >= 3:
        print("---DECISION: MAX QUERY TRANSFORMATIONS REACHED (2), GENERATE ANYWAY---")
        return "generate"
    
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    # Get counters with defaults
    transform_count = state.get("transform_count", 0)
    generation_attempts = state.get("generation_attempts", 0)
    
    # CRITICAL: Safety check to prevent infinite loops
    # If we've tried generating too many times, just end
    if generation_attempts >= 3:
        print(f"---DECISION: MAX GENERATION ATTEMPTS REACHED ({generation_attempts}), ENDING---")
        return "useful"  # Force end after too many generation attempts

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            # Check if we've reached max transformations
            if transform_count >= 2:
                print("---DECISION: MAX QUERY TRANSFORMATIONS REACHED (2), END ANYWAY---")
                return "useful"  # Force end after max transformations
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        # Check if we've reached max transformations or generations
        if transform_count >= 2 or generation_attempts >= 2:
            print("---DECISION: MAX ATTEMPTS REACHED, END ANYWAY---")
            return "useful"  # Force end after max attempts
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    


def get_self_rag_graph():
    from langgraph.graph import END, StateGraph, START

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile with explicit recursion limit
    app = workflow.compile()  # Increased from default 25

    return app