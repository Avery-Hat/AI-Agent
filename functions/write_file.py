import os

# def write_file(working_directory, file_path, content):
#     try:
#         abs_work = os.path.abspath(working_directory)
#         abs_target = os.path.abspath(os.path.join(working_directory, file_path))

#         # 1. Block absolute paths
#         if os.path.isabs(file_path):
#             return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'
#         # 2. Block directory traversal outside working_directory
#         if os.path.commonpath([abs_target, abs_work]) != abs_work:
#             return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'
#         # Ensure parent directories exist
#         parent = os.path.dirname(abs_target)
#         if parent:
#             os.makedirs(parent, exist_ok=True)

#         # Overwrite or create the file
#         with open(abs_target, "w", encoding="utf-8") as f:
#             f.write(content)

#         return f'Successfully wrote to "{file_path}" ({len(content)} characters written)'

#     except Exception as e:
#         return f"Error: {e}"

import os


def write_file(working_directory, file_path, content):
    abs_working_dir = os.path.abspath(working_directory)
    abs_file_path = os.path.abspath(os.path.join(working_directory, file_path))
    if not abs_file_path.startswith(abs_working_dir):
        return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'
    if not os.path.exists(abs_file_path):
        try:
            os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        except Exception as e:
            return f"Error: creating directory: {e}"
    if os.path.exists(abs_file_path) and os.path.isdir(abs_file_path):
        return f'Error: "{file_path}" is a directory, not a file'
    try:
        with open(abs_file_path, "w") as f:
            f.write(content)
        return (
            f'Successfully wrote to "{file_path}" ({len(content)} characters written)'
        )
    except Exception as e:
        return f"Error: writing to file: {e}"