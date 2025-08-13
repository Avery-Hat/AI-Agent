import os

def get_files_info(working_directory, directory="."):
    try:
        # Build full path from working_directory + directory argument
        full_path = os.path.join(working_directory, directory)
        abs_working_dir = os.path.abspath(working_directory)
        abs_target_dir = os.path.abspath(full_path)

        # Guardrail: ensure target is inside working_directory
        if not abs_target_dir.startswith(abs_working_dir):
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'

        # Check if target is a directory
        if not os.path.isdir(abs_target_dir):
            return f'Error: "{directory}" is not a directory'

        # Get directory contents
        items = os.listdir(abs_target_dir)
        lines = []
        for item in items:
            item_path = os.path.join(abs_target_dir, item)
            try:
                size = os.path.getsize(item_path)
                is_dir = os.path.isdir(item_path)
                lines.append(f"- {item}: file_size={size} bytes, is_dir={is_dir}")
            except Exception as e:
                # Error getting item details — include in output
                lines.append(f"- {item}: Error getting info: {e}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"

