
from typing import List, Literal, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, StateGraph, START
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import numpy as np

# Define state for application
class State(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retrieved_count: int
    original_documents: List[Document]


class AdaptiveKSelector:
    """
    Faithful implementation of Adaptive-k selection strategy (Taguchi et al., 2025).
    Translated to NumPy to avoid PyTorch dependency.
    """
    def __init__(
        self,
        strategy: Literal["largest_gap", "moving_avg", "2diff_spike"] = "largest_gap",
        window: Optional[int] = None,          # for moving_avg
        retrieve_more: Optional[int | float] = 5,  # buffer B (int) or multiplier (float)
        ignore_extreme: float | int = 0.0,     # cut head
        ignore_extreme_tail: float | int = 0.0,# cut tail
        ignore_below_median: bool = False,
        min_k: int = 1,
        max_k: Optional[int] = None,
    ):
        self.strategy = strategy
        self.window = window
        self.retrieve_more = retrieve_more
        self.ignore_extreme = ignore_extreme
        self.ignore_extreme_tail = ignore_extreme_tail
        self.ignore_below_median = ignore_below_median
        self.min_k = min_k
        self.max_k = max_k

    def _convert(self, scores: List[float] | np.ndarray) -> np.ndarray:
        if isinstance(scores, list):
            arr = np.array(scores, dtype=np.float32)
        elif isinstance(scores, np.ndarray):
            arr = scores.astype(np.float32)
        else:
            raise TypeError("scores must be list or np.ndarray")

        # Ensure sorted descending
        # Check if sorted ascending
        if np.all(np.diff(arr) >= 0):
            arr = np.flip(arr)
        elif np.all(np.diff(arr) <= 0):
            pass  # already correct
        else:
            # sort descending
            arr = np.sort(arr)[::-1]
        return arr

    def _get_cut_indices(self, n: int) -> Tuple[int, int]:
        if self.ignore_below_median:
            self.ignore_extreme_tail = 0.5

        if isinstance(self.ignore_extreme, float):
            head = int((n - 1) * self.ignore_extreme)
        else:
            head = int(self.ignore_extreme)

        if isinstance(self.ignore_extreme_tail, float):
            tail = int((n - 1) * self.ignore_extreme_tail)
        else:
            tail = int(self.ignore_extreme_tail)
        return head, tail

    def _find_threshold_largest_gap(self, arr: np.ndarray) -> int:
        n = len(arr)
        head, tail = self._get_cut_indices(n)
        
        # Calculate gaps (descending, so diff is negative, we want largest drop)
        # We use adjacent difference: arr[i+1] - arr[i]
        # But standard logic uses gap size. 
        # We will follow the simple logic: largest drop between i and i+1
        # This corresponds to minimizing the (negative) diff, or maximizing (arr[i] - arr[i+1])
        
        # np.diff(arr) gives [arr[1]-arr[0], arr[2]-arr[1], ...]
        # Since arr is descending, these are negative.
        # The "largest gap" corresponds to the most negative value (largest magnitude drop)
        
        gaps = np.diff(arr)
        
        if tail == 0:
            sub = gaps[head:]
            offset = head
        else:
            sub = gaps[head:-tail]
            offset = head
            
        if len(sub) == 0:
            return len(arr) - 1

        # argmin of gaps == largest negative drop
        idx = np.argmin(sub)
        return idx + offset

    def find_threshold(self, scores: List[float] | np.ndarray) -> int:
        arr = self._convert(scores)
        n = len(arr)
        if n < 2:
            return 0  # at least one document

        # Defaulting to largest_gap for this implementation as it's the most robust default
        # and doesn't require tuning window sizes for moving_avg
        th = self._find_threshold_largest_gap(arr)
        return th

    def select_k(self, scores: List[float]) -> int:
        if not scores:
            return self.min_k

        threshold_idx = self.find_threshold(scores)  # index in [0, n-2]
        # threshold_idx corresponds to the gap between doc[idx] and doc[idx+1]
        # So we keep documents up to idx (inclusive), which is idx+1 docs.
        k = threshold_idx + 1

        # buffer (retrieve_more)
        if self.retrieve_more:
            if isinstance(self.retrieve_more, float):
                k = int(k * self.retrieve_more)
            elif isinstance(self.retrieve_more, int):
                k = k + self.retrieve_more

        # caps
        n = len(scores)
        if self.max_k is not None:
            k = min(k, self.max_k)
        
        k = max(self.min_k, min(k, n))
        return k


# Initialize Selector
selector = AdaptiveKSelector(
    strategy="largest_gap",
    retrieve_more=5,               # B = 5 extra docs buffer
    ignore_extreme=0.0,            
    ignore_extreme_tail=0.0,
    ignore_below_median=False,
    min_k=1,
    max_k=50,                      # matches candidate pool
)


# Define application steps
def retrieve(state: State):
    print("---ADAPTIVE RETRIEVE (Improved)---")
    question = state["question"]
    
    # Retrieve a large pool of candidates to analyze the distribution
    k_candidates = 50
    vector_store = PineconeVectorStore(
        index_name="seal-2wiki-v1", 
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    
    results = vector_store.similarity_search_with_score(question, k=k_candidates)
    
    if not results:
        return {"documents": [], "retrieved_count": 0, "original_documents": []}
        
    scores = [score for doc, score in results]
    k_optimal = selector.select_k(scores)
    
    original_docs = []
    final_docs = []
    
    for i, (doc, score) in enumerate(results):
        doc.metadata["score"] = float(score)
        original_docs.append(doc)
        if i < k_optimal:
            final_docs.append(doc)
    
    print(f"Adaptive-k: Retrieved {k_candidates} candidates -> Selected k={k_optimal}")
    print(f"Top Score: {scores[0]:.4f}, Cut-off Score: {scores[k_optimal-1]:.4f}")
    if k_optimal < len(scores):
        print(f"Next Score (Excluded): {scores[k_optimal]:.4f}, Gap: {scores[k_optimal-1] - scores[k_optimal]:.4f}")
    
    return {
        "documents": final_docs, 
        "retrieved_count": k_optimal,
        "original_documents": original_docs
    }


# Custom RAG prompt - same as basic_rag for fair comparison
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


def generate(state: State):
    print("---GENERATE---")
    docs_content = "\n\n".join(doc.page_content for doc in state["documents"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)
    return {"generation": response.content}


def get_adaptive_rag_graph():
    # Compile application
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)
    
    graph = graph_builder.compile()

    return graph
