import os
from dotenv import load_dotenv
from src.workflow_manager import build_seal_rag_graph
from src.other_rags.self_rag import get_self_rag_graph

load_dotenv()

# Default graph used by langgraph.json key "ETR:2.0" (./app.py:graph)
graph = build_seal_rag_graph()

# Graph used by langgraph.json key "self-rag" (./app.py:graph_self_rag)
graph_self_rag = get_self_rag_graph()


print("done")