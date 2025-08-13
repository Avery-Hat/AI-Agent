import os
from functions.config import MAX_FILE_CHARS

def get_file_content(working_directory, file_path):
    try:
        # Build the absolute paths
        abs_working_dir = os.path.abspath(working_directory)
        abs_target_file = os.path.abspath(os.path.join(working_directory, file_path))

        # Guardrail: file must be inside working_directory
        if not abs_target_file.startswith(abs_working_dir):
            return f'Error: Cannot read "{file_path}" as it is outside the permitted working directory'

        # Check if it's a regular file
        if not os.path.isfile(abs_target_file):
            return f'Error: File not found or is not a regular file: "{file_path}"'

        # Read file content
        with open(abs_target_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Truncate if too long
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + f'[...File "{file_path}" truncated at {MAX_FILE_CHARS} characters]'

        return content

    except Exception as e:
        return f"Error: {e}"
