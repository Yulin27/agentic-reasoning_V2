from tavily import TavilyClient
from nano_graphrag import GraphRAG


class ResearchAgent:
    """
    A research agent that can retrieve information from a language model
    and store relevant details into a knowledge graph.
    """

    def __init__(
        self,
        name: str,
        tavily_client: TavilyClient,
        knowledge_graph: GraphRAG,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> None:
        self.name = name
        self.tavily_client = tavily_client
        self.knowledge_graph = knowledge_graph
        self.top_k = top_k
        self.threshold = threshold

    def gather_information(self, query: str) -> str:
        """
        Obtain information from a Large Language Model (LLM).
        """
        print(f"[{self.name}] Gathering information for query: {query}")
        response = self.tavily_client.search(query, include_raw_content=True)
        response = response["results"][: self.top_k]
        for result in response:
            if result["score"] < self.threshold:
                response.remove(result)

        return response

    def analyze_and_store(self, data: str) -> None:
        """
        Perform analysis of the retrieved data and store relevant findings
        into the knowledge graph. Implementation of how your data is parsed,
        summarized, or broken down into graph relations is entirely up to you.
        """
        print(f"[{self.name}] Analyzing data:\n{data}")

        # Analyze the data and store relevant details into the knowledge graph
        # TODO: Implement the analysis logic here

        if self.knowledge_graph is not None:
            self.knowledge_graph.insert_data(data)
            print(f"[{self.name}] Stored data in KnowledgeGraph")

    def run(self, query: str):
        """
        The main loop for the agent if you want a single method that performs
        both gather + analyze/store steps.
        """
        info = self.gather_information(query)
        self.analyze_and_store([i["content"] for i in info])
        return info
