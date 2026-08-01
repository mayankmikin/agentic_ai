import io


def execute_python_code(code: str) -> str:
    """Executes a snippet of Python code and returns whatever was printed to stdout.

    This is a learning-only sandbox. exec() is fundamentally unsafe; do not expose this
    tool to untrusted users or networks. The restrictions below stop the casual cases,
    not a determined attacker.
    """
    try:
        # A minimal restricted environment. We strip __builtins__ down to a small
        # whitelist so that, e.g., open(), eval(), and __import__ are not directly
        # available from the snippet's global scope.
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
            "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
            "pow": pow, "print": print, "range": range, "repr": repr, "reversed": reversed,
            "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "zip": zip,
        }
        # Pre-import a couple of safe, useful modules so the model doesn't have to
        import math, statistics
        restricted_globals = {
            "__builtins__": safe_builtins,
            "math": math,
            "statistics": statistics,
        }

        # Capture stdout so we can hand the printed output back to the model
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, restricted_globals, {})

        output = buffer.getvalue().strip()
        if not output:
            return "Code executed successfully but produced no output. Use print() to return a value."
        return f"Output:\n{output}"

    except Exception as e:
        return f"Execution error: {type(e).__name__}: {e}"