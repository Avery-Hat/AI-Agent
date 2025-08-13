import os
import sys
from google import genai
from dotenv import load_dotenv
from google.genai import types
from prompts import system_prompt  # Add this import


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Error, try again")
        sys.exit(1)

    # Parse arguments
    verbose = False
    prompt_parts = []

    for arg in sys.argv[1:]:
        if arg == "--verbose":
            verbose = True
        elif arg.startswith("--"):
            # Skip other potential flags in the future
            continue
        else:
            prompt_parts.append(arg)

    # Join non-flag arguments into a single prompt string
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
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_prompt),  # Add this line
    )

    if verbose:
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)

    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()