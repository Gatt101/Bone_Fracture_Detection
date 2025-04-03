# hospital_locator.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")  # Store your API key in .env

def find_nearby_orthopedic_hospitals(lat, lng, radius=5000):
    try:
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": "hospital",
            "keyword": "orthopedic",
            "key": GOOGLE_API_KEY
        }
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200 and data.get("results"):
            hospitals = []
            for place in data["results"]:
                hospitals.append({
                    "name": place.get("name"),
                    "address": place.get("vicinity"),
                    "rating": place.get("rating"),
                    "location": place.get("geometry", {}).get("location"),
                    "place_id": place.get("place_id")
                })
            return hospitals
        else:
            return []
    except Exception as e:
        return {"error": str(e)}
