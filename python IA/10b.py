import json
with open('weather_data.json')as f:
    data = json.load(f)
curr_temp = data['main']['temp']
humidity = data['main']['humidity']
weather_desc = data['weather'][0]['weather description']
print(f"Current Temparature : {curr_temp}")
print(f"Humidity : {humidity}%")
print(f"Weather Description : {weather_desc}")