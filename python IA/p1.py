import json
with open('weather_data.json') as f:
    data = json.load(f)
curtemp = data['main']['temp']
humidity = data['main']['humidity']
weather_desc = data['weather'][0]['weather description']
print(f"Cur temp :{curtemp}^C")
print(f"Humidity : {humidity}%")
print(f"Weather Description : {weather_desc}")