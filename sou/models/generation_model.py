from litellm import completion


class Model:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def generate_response(
        self,
        messages: list,
        temperature: int = 0.7,
        top_p: float = None,
        max_tokens: int = None,
        stop_tokens: list = None,
        frequency_penalty: float = None,
    ) -> str:
        response = completion(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            frequency_penalty=frequency_penalty,
        )
        return response.choices[0].message.content
