from litellm import embedding


class EmbeddingModel:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name

    def generate_embedding(self, input_text: str) -> list:
        response = embedding(model=self.model_name, input=input_text)
        return response["data"][0]["embedding"]

    def generate_embeddings(self, input_texts: list) -> list:
        response = embedding(model=self.model_name, input=input_texts)
        return [item["embedding"] for item in response["data"]]
