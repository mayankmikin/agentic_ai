import sys
import os
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
