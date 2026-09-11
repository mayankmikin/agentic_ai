from fastmcp import FastMCP

from app import get_books

# Initialize the server 
mcp = FastMCP("Weather Server")


@mcp.tool()
def get_weather(location: str) -> dict:
    """Gets current weather for a location."""
    return {
        "temperature": 72.5,
        "conditions": "Sunny",
        "location": location,
    }


@mcp.tool()
def get_forecast(location: str, days: int = 1) -> dict:
    """Gets weather forecast for a location for the specified number of days."""
    return {
        "location": location,
        "forecast": [
            {
                "day": i + 1,
                "temperature": 70 + i,
                "conditions": "Partly Cloudy",
            }
            for i in range(days)
        ],
    }

@mcp.tool()
def list_books() -> list[dict]:
    """Gets books from SQLite DB."""
    return get_books()

if __name__ == "__main__":
    # Runs the server using standard input/output transport
    mcp.run(transport="stdio")