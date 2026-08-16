# The tools_local defined in the Ollama JSON schema format
AVAILABLE_TOOLS =[
    {
        "type": "function",
        "function": {
            "name": "list_directory_contents",
            "description": (
                "Lists files and subdirectories inside a path within the user's workspace. "
                "Use this to inspect the environment before answering questions about local files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "A relative path inside the workspace, e.g. '.', 'data', or 'src/utils'. "
                            "Defaults to the workspace root."
                        )
                    }
                },
                "required": []
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Gets the current temperature for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. Tokyo"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city"]
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": (
                "Runs a short Python snippet and returns whatever it prints to stdout. "
                "Use this for precise arithmetic, string manipulation, or any logic you "
                "would otherwise have to guess at. The snippet must use print() to return a value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "A self-contained Python snippet. The 'math' and 'statistics' "
                            "modules are pre-imported. Always call print() on the final value."
                        )
                    }
                },
                "required": ["code"]
            }
        }
    }
]