# extract the research query and code query
from sou.prompt.utils.extract_pattern import extract_between, extract_boxed
from sou.agents.code_agent import CodeAgent
from sou.agents.graphrag_agent import GraphRAGAgent
from sou.agents.research_agent import ResearchAgent
from sou.reasoning.run_generation import run_generation
from sou.prompt.config import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_CODE_QUERY,
    END_CODE_QUERY,
    BEGIN_MIND_MAP_QUERY,
    END_MIND_MAP_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
    BEGIN_CODE_RESULT,
    END_CODE_RESULT,
    BEGIN_MIND_MAP_RESULT,
    END_MIND_MAP_RESULT,
)
import litellm
import os


def get_mcq_answer(question: str) -> str:
    prompt = f"""
        You will be given a multiple-choice question about reliability engineering. 
        Choose the correct answer from the options provided. 
        Respond only with a single character: a, b, c, or d. No explanations.
        
        {question}
        """

    response = litellm.completion(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        temperature=0.0,
    )

    return response["choices"][0]["message"]["content"].strip()


# ReasoningSettgs BaseModel
# AgentList : List[Agent]
# Model:


def run_reasoning(
    filtered_data: list,
    active_sequences: list,
    forcing_research: bool,
    general_model_args: dict,
    code_mode_name: str = "gpt-4o",
    tokenizer: object = None,
    tavily_client: object = None,
    verbose: bool = False,
) -> list:
    """
    Run reasoning on the filtered data.
    """

    for i in range(len(filtered_data)):
        initial_prompt = filtered_data[i]["Question"]

        knowledge_graph_agent = GraphRAGAgent(
            name="GraphRAGAgent", working_dir=f"output/knowledge_graph-{i}"
        )

        research_agent = ResearchAgent(
            name="ResearchAgent",
            tavily_client=tavily_client,
            knowledge_graph=knowledge_graph_agent,
        )

        code_agent = CodeAgent(
            model_name=code_mode_name, api_key=os.environ["OPENAI_API_KEY"]
        )

        print(f"========= Sequence {i} =========")
        print(f"Initial Prompt: {initial_prompt}")

        # force the research agent to gather information and store it in the knowledge graph
        if forcing_research:
            try:
                search_context = research_agent.run(initial_prompt)
                if verbose:
                    print(f"Search Context: {search_context}")
            except Exception:
                search_context = ""
                print("Research Agent failed")
                pass

        sequence = active_sequences[i]
        turn = 0

        while not sequence["finished"]:
            turn += 1
            print(f"========= Sequence {i} Turn {turn} ========")

            # Summarize the context of the initial prompt
            try:
                context = knowledge_graph_agent.summary_context(initial_prompt)
                if verbose:
                    print(f"========= Context: {context} ========")
                    print("====================")

            except Exception:
                context = ""
                print("Knowledge Graph Agent failed")
                pass

            # Append context to prompt
            sequence["prompt"] += "Context: " + context + "\n\n"

            # Generate response
            output = run_generation(
                prompt=sequence["prompt"],
                tokenizer=tokenizer,
                **general_model_args,
            )

            sequence["history"].append(output)
            # Append generated text to prompt and output
            sequence["prompt"] += output
            sequence["output"] += output

            # Insert data into the knowledge graph
            try:
                knowledge_graph_agent.insert_data(output)
            except Exception:
                print("Knowledge Graph Insertion Failed")
                pass

            # Extract the research query, code query, and mind map query
            research_query = extract_between(
                output, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY
            )
            code_query = extract_between(output, BEGIN_CODE_QUERY, END_CODE_QUERY)
            mind_map_query = extract_between(
                output, BEGIN_MIND_MAP_QUERY, END_MIND_MAP_QUERY
            )

            # Mark sequence as complete if no queries are found
            if research_query is None and code_query is None and mind_map_query is None:
                sequence["finished"] = True
                sequence["final_result"] = extract_boxed(output)
                # If the final result is not in the format a, b, c, or d, ask the llm to provide the answer, sure gpt-4o is overkill for this,  it's just for demonstration purposes
                if sequence["final_result"] is not None and sequence[
                    "final_result"
                ] not in ["a", "b", "c", "d"]:
                    print(
                        f"Final result is not in the correct format: {sequence['final_result']}"
                    )
                    result = sequence["final_result"]
                    sequence["final_result"] = get_mcq_answer(result)
                print(f"Final Result: {sequence['final_result']}")
                print("Sequence marked as complete.")

            # Run the research agent
            if research_query is not None:
                analysis = research_agent.run(research_query)
                append_text = (
                    f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                )
                sequence["prompt"] += append_text
                sequence["output"] += append_text
                sequence["history"].append(append_text)
                print(f"Research Output {i}: {analysis}")

            # Run the code agent based on the query and context from the knowledge graph if available
            if code_query is not None:
                if knowledge_graph_agent is not None:
                    context = knowledge_graph_agent.summary_prev_reasoning(code_query)
                    if verbose:
                        print(f"-> Context from KG: {context}")
                else:
                    context = output
                context = output
                code_result = code_agent(code_query, context=output)
                append_text = (
                    f"\n\n{BEGIN_CODE_RESULT}{code_result}{END_CODE_RESULT}\n\n"
                )
                sequence["prompt"] += append_text
                sequence["output"] += append_text
                sequence["history"].append(append_text)
                print(f"Code Output {i}: {code_result}")

            # Run the mind map agent
            if mind_map_query is not None:
                mind_map_result = knowledge_graph_agent.answer_query(mind_map_query)
                append_text = f"\n\n{BEGIN_MIND_MAP_RESULT}{mind_map_result}{END_MIND_MAP_RESULT}\n\n"
                sequence["prompt"] += append_text
                sequence["output"] += append_text
                sequence["history"].append(append_text)
                if verbose:
                    print(f"Mind Map Output {i}: {mind_map_result}")

    return active_sequences
