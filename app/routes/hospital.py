from flask import Blueprint, request, jsonify
from app.utils.hospital_locator import find_nearby_orthopedic_hospitals

bp = Blueprint('hospital', __name__)

@bp.route("/nearby_hospitals", methods=["POST"])
def nearby_hospitals():
    try:
        data = request.get_json()
        lat = data.get("lat")
        lng = data.get("lng")

        if not lat or not lng:
            return jsonify({"error": "Latitude and longitude are required."}), 400

        hospitals = find_nearby_orthopedic_hospitals(lat, lng)
        return jsonify({"hospitals": hospitals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500 