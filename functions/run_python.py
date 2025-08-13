import os
import subprocess
import sys

def run_python_file(working_directory, file_path, args=[]):
    try:
        # Resolve real paths to protect against ../ and symlink escapes
        abs_work = os.path.realpath(os.path.abspath(working_directory))
        abs_target = os.path.realpath(os.path.abspath(os.path.join(working_directory, file_path)))

        # Guardrail: must be inside working_directory
        try:
            if os.path.commonpath([abs_target, abs_work]) != abs_work:
                return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'
        except ValueError:
            return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'

        # File existence check
        if not os.path.exists(abs_target):
            return f'Error: File "{file_path}" not found.'

        # Must be Python file
        if not file_path.endswith(".py"):
            return f'Error: "{file_path}" is not a Python file.'

        # Build command
        cmd = [sys.executable, abs_target] + list(args)

        # Run process
        completed = subprocess.run(
            cmd,
            cwd=abs_work,
            capture_output=True,
            text=True,
            timeout=30
        )

        # output
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        result_parts = []

        if stdout:
            result_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            result_parts.append(f"STDERR:\n{stderr}")
        if completed.returncode != 0:
            result_parts.append(f"Process exited with code {completed.returncode}")

        if not result_parts:
            return "No output produced."

        return "\n".join(result_parts)

    except Exception as e:
        return f"Error: executing Python file: {e}"
