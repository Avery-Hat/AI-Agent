import os
import sys
from google import genai
from dotenv import load_dotenv
from google.genai import types
from prompts import system_prompt

# If you already defined schemas in their own files, import them instead of redefining here:
from functions.get_files_info import schema_get_files_info

# --- New schemas (define here if you don't have separate schema variables) ---
schema_get_file_content = types.FunctionDeclaration(
    name="get_file_content",
    description="Reads and returns the contents of a file, constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the file, relative to the working directory.",
            ),
        },
        required=["file_path"],
    ),
)

schema_run_python_file = types.FunctionDeclaration(
    name="run_python_file",
    description="Executes a Python file with optional arguments, constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Python file to execute, relative to the working directory.",
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Optional list of arguments to pass to the Python file.",
            ),
        },
        required=["file_path"],
    ),
)

schema_write_file = types.FunctionDeclaration(
    name="write_file",
    description="Writes or overwrites content to a file, constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Destination file path, relative to the working directory.",
            ),
            "content": types.Schema(
                type=types.Type.STRING,
                description="Text content to write.",
            ),
        },
        required=["file_path", "content"],
    ),
)
# ---------------------------------------------------------------------------

# --- Single, top-level tool registry (what the grader expects) ---
available_functions = types.Tool(
    function_declarations=[
        schema_get_files_info,
        schema_get_file_content,
        schema_run_python_file,
        schema_write_file,
    ]
)
# ----------------------------------------------------------------

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
            system_instruction=system_prompt,   # make sure prompts.py lists all 4 ops
            tools=[available_functions],
        ),
    )

    if verbose and getattr(response, "usage_metadata", None):
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)

    # Print function calls if present; otherwise print text
    if getattr(response, "function_calls", None):
        for fc in response.function_calls:
            # fc.args prints like {'file_path': 'main.py'} / {'directory': 'pkg'}
            print(f"Calling function: {fc.name}({fc.args})")
    else:
        print("Response:")
        print(response.text)

if __name__ == "__main__":
    main()
