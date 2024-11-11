import requests
import datetime
import csv

def check_website_status(urls):
    results = []
    for url in urls:
        try:
            response = requests.get(url)
            status = "UP" if response.status_code == 200 else "DOWN"
        except:
            status = "DOWN"
            
        results.append({
            "url": url,
            "status": status,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Save results to CSV
    with open("website_status.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "status", "timestamp"])
        writer.writerows(results)

# Usage
websites = [
    "https://www.nischalskanda.xyz",
    "https://notfedex.000webhostapp.com/",
]
check_website_status(websites)