import json
import httpx
from openai import OpenAI
import json
import urllib.request
import urllib.parse

def get_current_weather(city: str, unit: str = "celsius") -> str:
    """Gets the current temperature for a given city using open-meteo API."""
    try:
        # Geocode the city to get latitude and longitude
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
        geo_req = urllib.request.Request(geo_url, headers={'User-Agent': 'Gemma4ToolCalling/1.0'})
        with urllib.request.urlopen(geo_req) as response:
            geo_data = json.loads(response.read().decode('utf-8'))

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find coordinates for city: {city}."

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        country = location.get("country", "")

        # Fetch the weather
        temp_unit = "fahrenheit" if unit.lower() == "fahrenheit" else "celsius"
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m&temperature_unit={temp_unit}"
        weather_req = urllib.request.Request(weather_url, headers={'User-Agent': 'Gemma4ToolCalling/1.0'})
        with urllib.request.urlopen(weather_req) as response:
            json_response=response.read().decode('utf-8')
            print(json_response)
            weather_data = json.loads(json_response)

        if "current" in weather_data:
            current = weather_data["current"]
            temp = current["temperature_2m"]
            wind = current["wind_speed_10m"]
            temp_unit_str = weather_data["current_units"]["temperature_2m"]
            wind_unit_str = weather_data["current_units"]["wind_speed_10m"]

            return f"The current weather in {city.title()} ({country}) is {temp}{temp_unit_str} with wind speeds of {wind}{wind_unit_str}."
        else:
            return f"Weather data for {city} is unavailable from the API."

    except Exception as e:
        return f"Error fetching weather for {city}: {e}"


# Point to Docker Model Runner's OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # required but not used
)


# 2. Instruct Gemma 4 on how to think and act
SYSTEM_PROMPT = """
You are a smart AI Agent operating in a loop: Reason, Act, Observe.
You have access to the following tools:
- get_current_weather(city: str, unit: str = "celsius"): Returns the latest weather update of the city in degree celsius.

To use a tool, you MUST use the exact JSON format below:
Action: {"tool": "tool_name", "args": {"param": "value"}}

When you have the final answer after observing tool outputs, reply with:
Final Answer: [Your definitive response here]

Always output your internal thought process first.
"""

# A dictionary to easily access the functions by name
AVAILABLE_TOOLS = {
    "get_current_weather": get_current_weather
}

def run_agent(user_request: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]

    # Limit execution loop to prevent infinite runaways
    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")

        # Trigger Gemma 4's reasoning mode natively
        response = client.chat.completions.create(
            model="gemma4:E4B",
            messages=messages,
            temperature=1.0, # Recommended sampling for Gemma 4
            extra_body={"chat-template-kwargs": {"enable_thinking": True}}
        )

        response_text = response.choices[0].message.content
        print(response_text)

        # Append the assistant's turn to memory
        messages.append({"role": "assistant", "content": response_text})

        # Scenario A: Agent reached a conclusion
        if "Final Answer:" in response_text:
            break

        # Scenario B: Agent is requesting a Tool execution
        if "Action:" in response_text:
            try:
                # Parse the Action JSON block
                action_line = [line for line in response_text.split('\n') if "Action:" in line][0]
                action_json = json.loads(action_line.replace("Action:", "").strip())

                tool_name = action_json["tool"]
                tool_args = action_json["args"]

                # Execute the bound python function
                if tool_name in AVAILABLE_TOOLS:
                    print(f"[System Execution]: Running {tool_name}({tool_args})...")
                    observation = AVAILABLE_TOOLS[tool_name](**tool_args)
                else:
                    observation = f"Error: Tool '{tool_name}' is not recognized."

                print(f"[System Observation]: {observation}")

                # Feed the feedback back into the loop as a new user interaction
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            except Exception as e:
                error_msg = f"Error parsing/executing action: {str(e)}. Correct your formatting."
                messages.append({"role": "user", "content": error_msg})


run_agent("what is the capital of norway and what is the temperature there right now?")