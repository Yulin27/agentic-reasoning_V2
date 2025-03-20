import os
import subprocess

import litellm


class CodeAgent:
    """Agent that generates Python code from a given query using a Language Model and executes it safely in Docker."""

    def __init__(
        self, model_name=None, api_key=None, docker_image_name="python_executor"
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.docker_image_name = docker_image_name
        self.build_docker_image()

    def build_docker_image(self):
        """Build the Docker image used for executing Python code."""
        dockerfile_content = """
        FROM python:3.11-slim
        WORKDIR /app
        """

        dockerfile_path = "temp/tools/Dockerfile"
        os.makedirs(os.path.dirname(dockerfile_path), exist_ok=True)

        with open(dockerfile_path, "w") as file:
            file.write(dockerfile_content)

        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                self.docker_image_name,
                "-f",
                dockerfile_path,
                "temp/tools",
            ],
            check=True,
        )

    def generate_code(self, query: str, context: str = "") -> str:
        """Generate Python code from a given query using a Language Model and execute it in Docker."""
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

        if "```python" in result:
            result = result[
                result.find("```python") + 9 : result.rfind("```")
            ]  # extract python code
        elif "```" in result:
            result = result[
                result.find("```") + 3 : result.rfind("```")
            ]  # extract general code

        path = "temp/tools/temp.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as file:
            file.write(result)

        return self.execute_code_in_docker(path)

    def execute_code_in_docker(self, script_path: str):
        """Run the Python script within a Docker container."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{os.path.abspath(script_path)}:/app/temp.py",
                    self.docker_image_name,
                    "python",
                    "temp.py",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return result.stdout

        except subprocess.TimeoutExpired:
            return "Code execution timed out after 10 seconds"

        except subprocess.CalledProcessError as e:
            return f"Execution error: {e.stderr}"

    def __call__(self, query: str, context: str = "") -> str:
        """Convenience method to directly generate and execute Python code."""
        return self.generate_code(query, context)


# Example usage:
# code_agent = CodeAgent(model_name="gpt-3.5-turbo", api_key = os.environ["OPENAI_API_KEY"])
# query = "Find the sum of all even numbers from 1 to 100."
# context = "You are given a range of numbers from 1 to 100."
# output = code_agent(query, context)
# print(output)
