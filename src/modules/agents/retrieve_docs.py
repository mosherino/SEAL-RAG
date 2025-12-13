### Retrieve docs 
from langgraph.constants import Send

from src.states import State
from src.modules.configuration import Configuration
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

import asyncio


def get_document_title(document: Document) -> str:
    """Return a stable title key for a Document.

    Priority order:
    - metadata['title'] / 'Title' / 'name' / 'source'
    - fallback to metadata['id']
    - fallback to first 80 chars of page_content
    """
    metadata = document.metadata or {}
    title = metadata.get("title") or metadata.get("Title") or metadata.get("name") or metadata.get("source")
    if isinstance(title, str) and title.strip():
        return title.strip()
    fallback = metadata.get("id") or document.page_content[:80]
    return str(fallback).strip()


def merge_documents_by_title(existing_documents: list[Document], new_documents: list[Document]) -> list[Document]:
    """Merge two document lists, removing duplicates by title.

    A document is considered a duplicate if its resolved title (via get_document_title)
    matches a previously seen title.
    """
    merged: list[Document] = []
    seen_titles: set[str] = set()
    for doc in existing_documents + new_documents:
        key = get_document_title(doc)
        if key and key not in seen_titles:
            seen_titles.add(key)
            merged.append(doc)
    return merged


async def retrieve_docs(state: State, config: RunnableConfig) -> State:
    """
    Retrieve documents and update state

    Args:
        state (dict): The current graph state
        config: Runtime configuration

    Returns:
        State: Updated state with retrieved documents
    """
    print("---RETRIEVE---")
    # Get the configuration from the config dictionary
    configurable = config.get("configurable", {})
    # Create a Configuration object from the configurable dictionary
    configuration = Configuration(**configurable)
    # Build the retriever
    user_query = state.get("micro_query") or state["user_query"]
    
    # Run the synchronous builder in a separate thread to avoid blocking the loop
    
    retriever = await asyncio.to_thread(configuration.build_retriever)
    
    # Retrieval
    new_docs: list[Document] = await retriever.ainvoke(user_query) or []

    # Merge with existing and deduplicate by title (with fallbacks)
    existing_docs: list[Document] = state.get("documents", []) or []
    # Compute new (by title) and merged
    existing_titles = {get_document_title(d) for d in existing_docs}
    docs_new_only = [d for d in new_docs if get_document_title(d) not in existing_titles]
    merged = merge_documents_by_title(existing_docs, new_docs)

    # Return updated state
    return {
        "documents": merged, 
        "documents_new": docs_new_only,
        "repair_loop_limit": configuration.repair_loop_limit
    }



def continue_to_cached_entities_update_agent(state: State):
    """
    Simple dispatch to cached entities update.
    
    Args:
        state: Current state containing documents and user query
        
    Returns:
        List of Send commands for parallel cached entities extraction
    """
    print("---CONTINUE TO CACHED ENTITIES UPDATE AGENT---")
    
    user_query = state["user_query"]
    documents = state.get("documents_new") or []
    micro_query = state.get("micro_query", None)
    cached_entities = state.get("cached_entities", [])
    repair_decision = state.get("repair_decision", None)
    
    print(f"Dispatching {len(documents)} documents for cached entities extraction")
    print(f"Original query: {user_query}")
    print(f"Micro query: {micro_query}")
    
    if not documents:
        print("WARNING: No documents to extract entities from!")
        # Decide where to go next based on loop count. If we haven't hit the
        # loop limit, try generating another micro query. Otherwise, proceed to
        # repair decision using whatever we have so far.
        loop_count = state.get("repair_loop_count") or 0
        loop_limit = state.get("repair_loop_limit") or 3
        if loop_count < loop_limit:
            print(f"No new docs and loop_count={loop_count} < {loop_limit} → micro_query_agent")
            return "micro_query_agent"
        else:
            print(f"No new docs and loop_count={loop_count} ≥ {loop_limit} → to_repair")
            return "to_repair"
    
    send_commands = [Send("cached_entities_update", {
        "doc_to_extract": doc, 
        "user_query": user_query, 
        "micro_query": micro_query,
        "cached_entities": cached_entities or [],  # Pass empty list if None
        "repair_decision": repair_decision
    }) for doc in documents]
    
    print(f"Created {len(send_commands)} Send commands for parallel processing")
    return send_commands