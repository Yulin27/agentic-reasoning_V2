# Hypothetical imports (adjust based on your actual package/module structure):
from nano_graphrag import GraphRAG


class GraphRAGAgent:
    def __init__(self, name: str, working_dir: str) -> None:
        self.name = name
        self.kg = GraphRAG(working_dir)

    def answer_query(self, query: str, store_result: bool = True) -> str:
        print(
            f"[{self.name}] Retrieving context from knowledge graph for answering query: {query}"
        )
        return self.kg.query(query)

    def summary_prev_reasoning(self, query: str) -> str:
        print(f"[{self.name}] Retrieving context from knowledge graph for: {query}")
        return self.kg.query(
            f"Summarize the reasoning process of this query: {query}, be short and clear."
        )

    def summary_context(self, query: str) -> str:
        # print(f"[{self.name}] Retrieving context from knowledge graph for: {query}")
        return self.kg.query(
            f"Summarize the context of this query: {query}, be short and clear, for a human to understand better the context."
        )

    def insert_data(self, data: str) -> None:
        print(f"[{self.name}] Inserting data into knowledge graph")
        self.kg.insert(data)
