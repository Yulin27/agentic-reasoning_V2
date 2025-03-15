import subprocess
import litellm
import os


class CodeAgent:
    """Agent that generates Python code from a given query using a Language Model."""

    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name
        self.api_key = api_key

    def generate_code(self, query: str, context: str = "") -> str:
        """Generate Python code from a given query using a Language Model and execute it
        to return the output."""
        # Generate code using LLM
        prompt = f"Given the Context: {context}\n\nWrite a Python script that solves the Problem. Ensure it can be run and outputs results directly. OUTPUT ONLY CODE. Problem: {query}"
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            api_key=self.api_key,
        )

        result = response["choices"][0]["message"]["content"]
        # Extract code from the result
        if "```python" in result:
            result = result[result.find("```python") + 9 : result.rfind("```")].strip()
        elif "```" in result:
            result = result[result.find("```") + 3 : result.rfind("```")].strip()

        # Save the code to a temporary file and execute it
        path = "temp/tools/temp.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as file:
            file.write(result)

        try:
            result = subprocess.run(
                ["python", path], capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Code execution timed out after 10 seconds"

    def __call__(self, query: str, context: str = "") -> str:
        """Generate Python code from a given query using a Language Model."""
        return self.generate_code(query, context)


# example usage
# code_agent = CodeAgent(model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
# query = "Find the sum of all even numbers from 1 to 100."
# context = "You are given a range of numbers from 1 to 100."
# code = code_agent(query, context)
# print(code)
