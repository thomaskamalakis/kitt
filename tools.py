"""Local tools the LLM can call: weather (Open-Meteo) and web search (DuckDuckGo)."""
import json

import requests

TIMEOUT = 10

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather and today's forecast for a place.",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string", "description": "City or area, e.g. 'Marousi, Athens'"}},
            "required": ["place"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current events, news or facts you do not know.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"}},
            "required": ["query"]}}},
]

STATUS = {"get_weather": "Checking the weather...", "web_search": "Searching the web..."}

WMO = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers",
       81: "showers", 82: "violent showers", 85: "snow showers", 86: "heavy snow showers",
       95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}


def _geocode(place):
    # Open-Meteo matches city names only: "Marousi, Athens" -> try the part before the comma first
    for name in dict.fromkeys([place.split(",")[0].strip(), place]):
        for lang in ("en", "el"):   # Greek names only match with language=el
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": name, "count": 1, "language": lang}, timeout=TIMEOUT)
            r.raise_for_status()
            results = r.json().get("results")
            if results:
                return results[0]
    return None


def get_weather(place):
    loc = _geocode(place)
    if not loc:
        return f"Could not find a place called '{place}'."
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=TIMEOUT, params={
        "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto", "forecast_days": 1,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
    r.raise_for_status()
    d = r.json()
    c, day = d["current"], d["daily"]
    where = ", ".join(x for x in (loc["name"], loc.get("admin1"), loc.get("country")) if x)
    return (f"Weather in {where} at {c['time'].replace('T', ' ')} local time: "
            f"{WMO.get(c['weather_code'], 'unknown conditions')}, {c['temperature_2m']:g} degrees Celsius "
            f"(feels like {c['apparent_temperature']:g}), humidity {c['relative_humidity_2m']:g} percent, "
            f"wind {c['wind_speed_10m']:g} kilometres per hour, precipitation {c['precipitation']:g} millimetres. "
            f"Today: high {day['temperature_2m_max'][0]:g}, low {day['temperature_2m_min'][0]:g} degrees Celsius, "
            f"chance of rain {day['precipitation_probability_max'][0]} percent.")


def web_search(query):
    from ddgs import DDGS
    # ddgs backends are flaky (rate limits / bot checks): try several, then the news
    # endpoint, and make a second pass before giving up.
    attempts = [lambda b=b: DDGS(timeout=6).text(query, max_results=5, backend=b)
                for b in ("auto", "bing", "yahoo", "duckduckgo")]
    attempts.append(lambda: DDGS(timeout=6).news(query, max_results=5))
    results = []
    for attempt in attempts * 2:
        try:
            results = attempt()
        except Exception:
            continue
        if results:
            break
    if not results:
        return "No search results were available."
    return "\n".join(f"- {x['title']}: {x.get('body') or x.get('source', '')}" for x in results)[:1500]


FUNCTIONS = {"get_weather": get_weather, "web_search": web_search}


def run(name, args):
    """Run a tool; failures come back as text so the model can explain them."""
    try:
        if isinstance(args, str):
            args = json.loads(args or "{}")
        return FUNCTIONS[name](**args)
    except KeyError:
        return f"Unknown tool '{name}'."
    except Exception as e:
        return f"The {name} tool failed: {type(e).__name__}."
