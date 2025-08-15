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

    # Start the conversation with the user’s message
    messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    # Agent loop: up to 20 steps
        # Agent loop: up to 20 steps
    for _ in range(20):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=messages,  # always send the full conversation
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[available_functions],
                ),
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            sys.exit(1)

        if verbose and getattr(response, "usage_metadata", None):
            print("Prompt tokens:", response.usage_metadata.prompt_token_count)
            print("Response tokens:", response.usage_metadata.candidates_token_count)

        # Safety: no candidates -> nothing we can do
        if not getattr(response, "candidates", None):
            print("No output produced.")
            break

        # Add ALL candidate contents to the transcript first
        # (This records the model's tool plan/function-call intents)
        had_function_call = False
        for cand in response.candidates:
            content = cand.content
            if content is None:
                continue

            # 1) Append the model's content (including function_call parts) to messages
            messages.append(content)

            # 2) Scan this content's parts for function calls
            for part in getattr(content, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if not fc:
                    continue
                had_function_call = True

                # Execute the tool call and append the tool response as a 'tool' message
                tool_msg = call_function(fc, verbose=verbose)
                # tool_msg is a types.Content(role="tool", parts=[from_function_response(...)]
                messages.append(tool_msg)

        # If the turn had *no* function calls, we consider it a final answer turn.
        if not had_function_call:
            # Prefer response.text, else join any text parts from the first candidate
            final_text = response.text
            if not final_text:
                first = response.candidates[0].content
                if first and getattr(first, "parts", None):
                    final_text = "\n".join(
                        p.text for p in first.parts if getattr(p, "text", None)
                    )
            if final_text:
                print("Final response:")
                print(final_text)
            else:
                print("No output produced.")
            break
    else:
        print("Reached iteration limit without a final response.")




if __name__ == "__main__":
    main()
