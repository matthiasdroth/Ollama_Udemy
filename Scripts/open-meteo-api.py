import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd  # only used for timestamp conversion

# Client with cache + retry
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Request hourly temperature for Berlin (52.52, 13.41)
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 47.7,
    "longitude": 9.2,
    "hourly": "temperature_2m",
}

response = openmeteo.weather_api(url, params=params)[0]
hourly = response.Hourly()

temps = hourly.Variables(0).ValuesAsNumpy()
times_start = hourly.Time()          # unix seconds
interval_s = hourly.Interval()       # seconds per step

# "Latest" = last value in the hourly series returned
latest_temp_c = float(temps[-1])
latest_time_utc = pd.to_datetime(times_start + (len(temps) - 1) * interval_s, unit="s", utc=True)

print(f"{latest_temp_c:.1f}°C at {str(latest_time_utc)[:10]} UTC")
