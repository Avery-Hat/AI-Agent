import os
from google.genai import types

def get_files_info(working_directory, directory="."):
    try:
        # Resolve real paths (handles .. and symlinks)
        abs_work = os.path.realpath(os.path.abspath(working_directory))
        abs_target = os.path.realpath(os.path.abspath(os.path.join(working_directory, directory)))

        # Guardrail: ensure target is inside working_directory
        try:
            if os.path.commonpath([abs_target, abs_work]) != abs_work:
                return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'
        except ValueError:
            # Different drives / roots -> definitely outside
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'

        # Check if target is a directory
        if not os.path.isdir(abs_target):
            return f'Error: "{directory}" is not a directory'

        # Get directory contents (sorted for stable output)
        items = sorted(os.listdir(abs_target))
        lines = []
        for item in items:
            item_path = os.path.join(abs_target, item)
            try:
                size = os.path.getsize(item_path)
                is_dir = os.path.isdir(item_path)
                lines.append(f"- {item}: file_size={size} bytes, is_dir={is_dir}")
            except Exception as e:
                lines.append(f'- {item}: Error getting info: {e}')

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


schema_get_files_info = types.FunctionDeclaration(
    name="get_files_info",
    description="Lists files in the specified directory along with their sizes, constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "directory": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The directory to list files from, relative to the working directory. "
                    "If not provided, lists files in the working directory itself."
                ),
            ),
        },
        # No 'required' => 'directory' is optional, which matches your function default
    ),
)
