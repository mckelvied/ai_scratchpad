import json
from typing_extensions import TypedDict
from agents import Agent, function_tool, Runner
import asyncio
from dotenv import load_dotenv
import requests
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

load_dotenv()

class Location(TypedDict):
    lat: float
    long: float

@function_tool  
async def fetch_weather(location: Location) -> str:
    """Fetch the weather for a given location.
    Args:
        location: The location to fetch the weather for.
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["lat"],
        "longitude": location["long"],
        "current": ["temperature_2m", "precipitation", "cloud_cover", "wind_speed_10m"],
        "forecast_days": 1,
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation: {response.Elevation()} m asl")
    # print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process current data. The order of variables needs to be the same as requested.
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_precipitation = current.Variables(1).Value()
    current_cloud_cover = current.Variables(2).Value()
    current_wind_speed_10m = current.Variables(3).Value()

    weather_info = f"""
Current time: {current.Time()}
Current temperature in celsius: {current_temperature_2m}
Current precipitation in millimeters: {current_precipitation}
Current cloud cover percentage: {current_cloud_cover}
Current wind speed km/h: {current_wind_speed_10m}
"""
    return weather_info
    


@function_tool  
async def get_my_location() -> Location:
    """Fetch the current location.
    """
    # Get location based on IP address using a free API
    response = requests.get("http://ip-api.com/json/?fields=lat,lon")
    data = response.json()
    return {"lat": data["lat"], "long": data["lon"]}
    

agent = Agent(
    name="Assistant",
    tools=[fetch_weather, get_my_location],  
)

async def main():
    result = await Runner.run(agent, "What is the weather at my location?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())