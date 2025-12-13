"""Define the configurable parameters for the agent."""

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

class Configuration(BaseModel):
    """Configuration for the agent and resource builders."""

    model: Annotated[
            Literal[
                "openai:gpt-4o-mini",
                "openai:gpt-4o",
                "openai:gpt-3.5-turbo",
                "openai:gpt-4.1",
                "openai:gpt-4.1-mini"
            ],
            {"__template_metadata__": {"kind": "llm"}},
        ] = Field(
            default="openai:gpt-4o",
            description="Provider:model, e.g. 'openai:gpt-4o'."
    )

    # LLM tunables
    temperature: Annotated[
        float,
        {"__template_metadata__": {"kind": "number", "min": 0.0, "max": 1.0}},
    ] = Field(
        default=0.0,
        description="Sampling temperature for the chat model.",
    )

    # Dedicated temperature for extraction (slightly warmer for recall)
    extraction_temperature: Annotated[
        float,
        {"__template_metadata__": {"kind": "number", "min": 0.0, "max": 1.0}},
    ] = Field(
        default=0.2,
        description="Sampling temperature used only for entity/relationship extraction.",
    )

    max_retries: Annotated[
        int,
        {"__template_metadata__": {"kind": "number", "min": 0, "max": 10}},
    ] = Field(
        default=3,
        description="Maximum retry attempts for LLM calls.",
    )

    retriever_k: Annotated[
        int,
        {"__template_metadata__": {"kind": "number", "min": 1, "max": 7}}
    ] = Field(
        default=5,
        description="The number of documents to retrieve from the vector store. Value between 1 and 7."
    )

    # Embedding model (OpenAI)
    embed_model: Annotated[
        Literal[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        {"__template_metadata__": {"kind": "select"}},
    ] = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use for vectorization.",
    )

    # Retriever selection and Pinecone settings
    retriever: Annotated[
        Literal[
            "pinecone"
        ],
        {"__template_metadata__": {"kind": "select"}},
    ] = Field(
        default="pinecone",
        description="The retrieval backend to use. Currently only 'pinecone' is supported.",
    )

    pinecone_index_name: Annotated[
        Literal[
            "seal-v3-hard",
            "seal-2wiki-v1" 
        ],
        {"__template_metadata__": {"kind": "select"}},
    ] = Field(
        default="seal-2wiki-v1",
        description="Pinecone index name to use for retrieval.",
    )

    pinecone_namespace: Annotated[
        Optional[str],
        {"__template_metadata__": {"kind": "text"}},
    ] = Field(
        default=None,
        description="Optional Pinecone namespace. Use None or empty to query the default namespace.",
    )

    # Repair loop behavior
    repair_loop_limit: Annotated[
        int,
        {"__template_metadata__": {"kind": "number", "min": 0, "max": 50}},
    ] = Field(
        default=1,
        description="Maximum number of micro-query iterations before forcing repair/answer.",
    )

    # ---- Builders ----

    def build_llm(self):
        """Build chat model via init_chat_model."""
        return init_chat_model(
            self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    def build_extraction_llm(self):
        """Build chat model for extraction with a slightly higher temperature."""
        return init_chat_model(
            self.model,
            temperature=self.extraction_temperature,
            max_retries=self.max_retries,
        )

    def build_retriever(self):
        """Build Pinecone retriever using OpenAI embeddings."""
        if self.retriever != "pinecone":
            raise ValueError(f"Unsupported retriever backend: {self.retriever}")
        embeddings = OpenAIEmbeddings(model=self.embed_model)
        vectorstore = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            embedding=embeddings,
            namespace=self.pinecone_namespace,
        )
        return vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})