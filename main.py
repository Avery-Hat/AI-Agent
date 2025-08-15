import os
import sys
from google import genai
from dotenv import load_dotenv
from google.genai import types
from prompts import system_prompt

# Import the shared tool registry (schemas) from functions/schemas.py
from functions.schemas import available_functions

# Import real tool implementations
from functions.get_files_info import get_files_info
from functions.get_file_content import get_file_content
from functions.write_file import write_file
from functions.run_python import run_python_file


def call_function(function_call_part, verbose: bool = False) -> types.Content:
    """
    Execute a tool call based on the model's FunctionCall.
    Returns a types.Content with a function_response Part whose payload is accessed at:
      content.parts[0].function_response.response  -> {"result": "..."} or {"error": "..."}
    """
    function_name = function_call_part.name
    args = dict(function_call_part.args or {})

    if verbose:
        print(f"Calling function: {function_name}({args})")
    else:
        print(f" - Calling function: {function_name}")

    # Map function name -> actual Python function
    registry = {
        "get_files_info": get_files_info,
        "get_file_content": get_file_content,
        "run_python_file": run_python_file,
        "write_file": write_file,
    }

    func = registry.get(function_name)
    if func is None:
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"error": f"Unknown function: {function_name}"},
                )
            ],
        )

    # Inject working directory so the LLM never controls it
    args["working_directory"] = "./calculator"

    try:
        result_str = func(**args)  # pass kwargs into the function
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"result": result_str},
                )
            ],
        )
    except Exception as e:
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"error": f"Execution failed: {e}"},
                )
            ],
        )


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Error, try again")
        sys.exit(1)

    # Parse flags & prompt
    verbose = False
    prompt_parts = []
    for arg in sys.argv[1:]:
        if arg == "--verbose":
            verbose = True
        elif arg.startswith("--"):
            continue
        else:
            prompt_parts.append(arg)

    prompt = " ".join(prompt_parts).strip()
    if not prompt:
        print("Error: No prompt provided.")
        sys.exit(1)

    if verbose:
        print(f"User prompt: {prompt}")

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[available_functions],
        ),
    )

    if verbose and getattr(response, "usage_metadata", None):
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)

    if getattr(response, "function_calls", None):
        for fc in response.function_calls:
            function_call_result = call_function(fc, verbose=verbose)

            # Validate expected payload shape
            try:
                payload = function_call_result.parts[0].function_response.response
            except Exception:
                raise RuntimeError("Fatal: tool call did not return a function_response payload")

            if verbose:
                print(f"-> {payload}")
    else:
        print("Response:")
        print(response.text)


if __name__ == "__main__":
    main()
