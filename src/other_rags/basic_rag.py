
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langgraph.graph import END, StateGraph, START
from typing_extensions import List, TypedDict
from langchain import hub



### Retrieval Grader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document



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


# Define state for application
class State(TypedDict):
    question: str
    documents: List[Document]
    generation: str


# Define application steps
def retrieve(state: State):

    vector_store = PineconeVectorStore(index_name="seal-2wiki-v1", embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vector_store.as_retriever(search_kwargs={"k": 50})#search_kwargs={"k": 3}
    retrieved_docs = retriever.invoke(state["question"])
    return {"documents": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["documents"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)
    return {"generation": response.content}


def get_basic_rag_graph():
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph