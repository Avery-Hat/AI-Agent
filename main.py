import os
import sys
from google import genai
from dotenv import load_dotenv
from google.genai import types
from prompts import system_prompt

# NEW: import the function declaration (schema)
from functions.get_files_info import schema_get_files_info


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Error, try again")
        sys.exit(1)

    # Parse arguments (keep your flexible flag handling)
    verbose = False
    prompt_parts = []
    for arg in sys.argv[1:]:
        if arg == "--verbose":
            verbose = True
        elif arg.startswith("--"):
            continue
        else:
            prompt_parts.append(arg)

    prompt = " ".join(prompt_parts)
    if not prompt.strip():
        print("Error: No prompt provided.")
        sys.exit(1)

    if verbose:
        print(f"User prompt: {prompt}")

    messages = [
        types.Content(role="user", parts=[types.Part(text=prompt)]),
    ]

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # NEW: register available tools (you can add more schemas later)
    available_functions = types.Tool(
        function_declarations=[
            schema_get_files_info,
        ]
    )

    print("Registered tools:", available_functions)

    # UPDATED: pass tools + your system prompt
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

    # NEW: if the model decided to call a function, show it; else show text
    if getattr(response, "function_calls", None):
        for fc in response.function_calls:
            print(f"Calling function: {fc.name}({fc.args})")
    else:
        print("Response:")
        print(response.text)


if __name__ == "__main__":
    main()
