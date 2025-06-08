import requests
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

def find_nearby_orthopedic_hospitals(lat, lng, radius=5000):
    """
    Find nearby orthopedic hospitals using Google Places API
    """
    try:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": "hospital",
            "keyword": "orthopedic",
            "key": GOOGLE_MAPS_API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data["status"] != "OK":
            return []

        hospitals = []
        for place in data.get("results", [])[:5]:  # Limit to 5 results
            hospital = {
                "name": place.get("name", ""),
                "address": place.get("vicinity", ""),
                "rating": place.get("rating", "N/A"),
                "location": place.get("geometry", {}).get("location", {}),
                "place_id": place.get("place_id", "")
            }
            hospitals.append(hospital)

        return hospitals

    except Exception as e:
        print(f"Error finding hospitals: {str(e)}")
        return [] 