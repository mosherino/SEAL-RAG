from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict
#from langgraph.graph.message import add_messages, MessagesState
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "DATE", "OTHER"]


class RepairDecisionOutput(BaseModel):
    """Output format for repair decision - whether we can answer the original query."""
    
    can_answer_original_query: str = Field(
        ...,
        description="Answer 'yes' if sufficient evidence exists to answer the original query, or 'no' if gaps remain"
    )
    reasoning: str = Field(
        ...,
        description="Provide a concise critical analysis: briefly state the key question(s) you investigated, your finding(s), and how this led to your decision. Keep it focused and direct."
    )

class SimpleTriplet(BaseModel):
    """Node1 --Edge--> Node2"""
    subject: str = Field(..., description="Source node (entity name)")
    subject_type: EntityType = Field(..., description="Source node type")
    relation: str = Field(..., description="Edge/relationship between nodes")
    object: str = Field(..., description="Target node (entity name)")
    object_type: EntityType = Field(..., description="Target node type")
    source_doc_title: str = Field(..., description="Title of Document where this triplet was found")
    confidence: int = Field(95, ge=0, le=100, description="Confidence score: 95-100 (explicit), 85-94 (clear), 70-84 (implied), 55-69 (inferred), 40-54 (weak), 25-39 (speculative), 0-24 (uncertain)")
    relevance: int = Field(50, ge=0, le=100, description="Relevance score: 95-100 (direct answer), 85-94 (key supporting fact), 70-84 (important context), 55-69 (useful detail), 40-54 (related), 25-39 (tangential), 0-24 (irrelevant)")


class StateInput(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        user_query: question
        generation: LLM generation
        documents: list of documents
    """
    user_query: str


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        user_query: question
        generation: LLM generation
        documents: list of documents
        readable_chains: relationship chains extracted for to_repair verification
        repair_decision: RepairDecisionOutput object containing decision and reasoning
    """
    user_query: str #Question to answer
    documents: List[Document] #Retrieved documents
    documents_new: Optional[List[Document]]
    relevance_documents: Optional[List[Document]]
    relevance_entities: Optional[List[SimpleTriplet]]
    micro_query: Optional[str]
    micro_query_history: Optional[List[str]]
    repair_loop_count: Optional[int]
    repair_loop_limit: Optional[int]
    cached_entities: Annotated[List[SimpleTriplet], operator.add]
    repair_decision: Optional[RepairDecisionOutput]  # Nested Pydantic object with decision + reasoning
    final_answer: Optional[str]  # LLM generated answer