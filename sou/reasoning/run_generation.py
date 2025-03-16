from typing import List, Dict
from sou.prompt.config import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_CODE_QUERY,
    END_CODE_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
    BEGIN_CODE_RESULT,
    END_CODE_RESULT,
    BEGIN_MIND_MAP_QUERY,
    END_MIND_MAP_QUERY,
    BEGIN_MIND_MAP_RESULT,
    END_MIND_MAP_RESULT,
)
import litellm


def run_generation(
    prompt: str,
    model: str,
    tokenizer,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int = 8192,
    stop_tokens: List[str] = None,
    frequency_penalty: float = 0.0,
    verbose: bool = False,
) -> List:
    """
    Run generation on a batch of sequences.

    Args:
        sequences: List of sequence dictionaries containing prompts
        llm: Language model instance
        tokenizer: Tokenizer instance
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens to generate
        stop_tokens: List of stop tokens
        frequency_penalty: Frequency penalty parameter

    Returns:
        List of generation outputs
    """

    if max_tokens is None:
        max_tokens = 8192

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_tokens,
        frequency_penalty=frequency_penalty,
    )
    generated_text = response["choices"][0]["message"]["content"]
    if verbose:
        print("Generated Text: ", generated_text)

    # Fix outputs that end with BEGIN_SEARCH_QUERY by adding END_SEARCH_QUERY
    if BEGIN_SEARCH_QUERY in generated_text and END_SEARCH_QUERY not in generated_text:
        generated_text = generated_text + END_SEARCH_QUERY
    if BEGIN_CODE_QUERY in generated_text and END_CODE_QUERY not in generated_text:
        generated_text = generated_text + END_CODE_QUERY
    if (
        BEGIN_MIND_MAP_QUERY in generated_text
        and END_MIND_MAP_QUERY not in generated_text
    ):
        generated_text = generated_text + END_MIND_MAP_QUERY

    return generated_text
