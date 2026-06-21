"""Weather lookup tool with simulated weather data for deterministic testing."""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool, ToolResult

# Hardcoded weather data for ~15 cities
_WEATHER_DATABASE: dict[str, dict[str, Any]] = {
    "san francisco": {
        "city": "San Francisco",
        "temperature_c": 18,
        "temperature_f": 64,
        "conditions": "Partly Cloudy",
        "humidity": 72,
        "wind_speed_kmh": 22,
        "wind_speed_mph": 14,
    },
    "new york": {
        "city": "New York",
        "temperature_c": 28,
        "temperature_f": 82,
        "conditions": "Sunny",
        "humidity": 55,
        "wind_speed_kmh": 12,
        "wind_speed_mph": 7,
    },
    "london": {
        "city": "London",
        "temperature_c": 15,
        "temperature_f": 59,
        "conditions": "Rainy",
        "humidity": 85,
        "wind_speed_kmh": 18,
        "wind_speed_mph": 11,
    },
    "tokyo": {
        "city": "Tokyo",
        "temperature_c": 32,
        "temperature_f": 90,
        "conditions": "Hot and Humid",
        "humidity": 78,
        "wind_speed_kmh": 8,
        "wind_speed_mph": 5,
    },
    "paris": {
        "city": "Paris",
        "temperature_c": 22,
        "temperature_f": 72,
        "conditions": "Clear",
        "humidity": 60,
        "wind_speed_kmh": 15,
        "wind_speed_mph": 9,
    },
    "sydney": {
        "city": "Sydney",
        "temperature_c": 14,
        "temperature_f": 57,
        "conditions": "Overcast",
        "humidity": 68,
        "wind_speed_kmh": 25,
        "wind_speed_mph": 16,
    },
    "mumbai": {
        "city": "Mumbai",
        "temperature_c": 35,
        "temperature_f": 95,
        "conditions": "Thunderstorms",
        "humidity": 90,
        "wind_speed_kmh": 30,
        "wind_speed_mph": 19,
    },
    "berlin": {
        "city": "Berlin",
        "temperature_c": 20,
        "temperature_f": 68,
        "conditions": "Partly Cloudy",
        "humidity": 62,
        "wind_speed_kmh": 14,
        "wind_speed_mph": 9,
    },
    "dubai": {
        "city": "Dubai",
        "temperature_c": 42,
        "temperature_f": 108,
        "conditions": "Scorching Hot",
        "humidity": 25,
        "wind_speed_kmh": 10,
        "wind_speed_mph": 6,
    },
    "toronto": {
        "city": "Toronto",
        "temperature_c": 25,
        "temperature_f": 77,
        "conditions": "Sunny",
        "humidity": 50,
        "wind_speed_kmh": 16,
        "wind_speed_mph": 10,
    },
    "beijing": {
        "city": "Beijing",
        "temperature_c": 30,
        "temperature_f": 86,
        "conditions": "Hazy",
        "humidity": 45,
        "wind_speed_kmh": 9,
        "wind_speed_mph": 6,
    },
    "moscow": {
        "city": "Moscow",
        "temperature_c": 12,
        "temperature_f": 54,
        "conditions": "Cloudy",
        "humidity": 75,
        "wind_speed_kmh": 20,
        "wind_speed_mph": 12,
    },
    "cairo": {
        "city": "Cairo",
        "temperature_c": 38,
        "temperature_f": 100,
        "conditions": "Hot and Dry",
        "humidity": 20,
        "wind_speed_kmh": 12,
        "wind_speed_mph": 7,
    },
    "singapore": {
        "city": "Singapore",
        "temperature_c": 31,
        "temperature_f": 88,
        "conditions": "Tropical Rain",
        "humidity": 88,
        "wind_speed_kmh": 11,
        "wind_speed_mph": 7,
    },
    "rio de janeiro": {
        "city": "Rio de Janeiro",
        "temperature_c": 26,
        "temperature_f": 79,
        "conditions": "Warm and Sunny",
        "humidity": 65,
        "wind_speed_kmh": 18,
        "wind_speed_mph": 11,
    },
}


class WeatherTool(BaseTool):
    """Get the current weather for a specified city (simulated)."""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get the current weather for a specified city."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get weather for.",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit preference.",
                    "default": "celsius",
                },
            },
            "required": ["city"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        city: str = kwargs.get("city", "")
        units: str = kwargs.get("units", "celsius")

        if not city.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="City name cannot be empty.",
            )

        city_key = city.strip().lower()
        weather = _WEATHER_DATABASE.get(city_key)

        if weather is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"City not found: '{city}'. Available cities include: {', '.join(d['city'] for d in _WEATHER_DATABASE.values())}",
            )

        # Format the response based on units
        if units == "fahrenheit":
            temperature = weather["temperature_f"]
            temp_unit = "°F"
            wind_speed = weather["wind_speed_mph"]
            wind_unit = "mph"
        else:
            temperature = weather["temperature_c"]
            temp_unit = "°C"
            wind_speed = weather["wind_speed_kmh"]
            wind_unit = "km/h"

        return ToolResult(
            tool_name=self.name,
            success=True,
            output={
                "city": weather["city"],
                "temperature": temperature,
                "unit": temp_unit,
                "conditions": weather["conditions"],
                "humidity": f"{weather['humidity']}%",
                "wind_speed": f"{wind_speed} {wind_unit}",
            },
        )
