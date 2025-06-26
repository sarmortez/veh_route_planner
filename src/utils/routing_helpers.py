import json
import math

# Constants
AVERAGE_BUS_SPEED_MPH = 30
BUS_CAPACITY = 15 # As per user clarification in the initial prompt

# Earth radius in miles (for Haversine distance)
EARTH_RADIUS_MILES = 3958.8

# Operating constraints
MAX_DAILY_OPERATION_HOURS = 8
STORE_WAIT_TIME_MINUTES = 40
MAX_ONE_WAY_PICKUP_TIME_MINUTES = 60 # 1 hour
# Total round trip time constraint: pickup + store_wait + dropoff <= 2h 40m (160 minutes)
# Pickup leg (store -> stops -> store) <= 1 hour
# Dropoff leg (store -> stops -> store) <= 1 hour
# So, total driving in one round (pickup leg + dropoff leg) <= 2 hours (120 minutes)
# Total round trip duration = driving_pickup + driving_dropoff + store_wait_time
# driving_pickup <= 60 min
# driving_dropoff <= 60 min (same route as pickup)
# Total round trip constraint: 60 (pickup) + 40 (wait) + 60 (dropoff) = 160 minutes (2 hours 40 minutes)


def parse_input_data(json_data_string):
    """Parses the JSON input string into a Python dictionary."""
    data = json.loads(json_data_string)
    # Round up demand for each transit stop
    for stop_id, stop_info in data.get("transitStops", {}).items():
        if "demand" in stop_info:
            stop_info["demand"] = math.ceil(stop_info["demand"])
            stop_info["weekly_demand_remaining"] = stop_info["demand"] # Initialize remaining demand
            stop_info["id"] = stop_id # Add id for easier access
    if "groceryStore" in data and data["groceryStore"]:
        store_id = list(data["groceryStore"].keys())[0]
        data["groceryStore"][store_id]["id"] = store_id
    return data

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees)."""
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_MILES * c
    return distance

def calculate_travel_time_minutes(distance_miles):
    """Calculate travel time in minutes given distance in miles and average speed in mph."""
    if AVERAGE_BUS_SPEED_MPH <= 0:
        return float('inf') # Avoid division by zero or negative speed
    time_hours = distance_miles / AVERAGE_BUS_SPEED_MPH
    time_minutes = time_hours * 60
    return time_minutes

# Sample data provided by the user
sample_data_json = """
{
  "transitStops": {
    "cts120970026": {
      "longitude": -81.4718,
      "latitude": 28.27864,
      "demand": 78.56256620049982
    },
    "cts120970019": {
      "longitude": -81.4589,
      "latitude": 28.25684,
      "demand": 113.25817428246684
    },
    "cts120970017": {
      "longitude": -81.4625,
      "latitude": 28.23921,
      "demand": 41.29263843246873
    },
    "cts120970009": {
      "longitude": -81.4666,
      "latitude": 28.21828,
      "demand": 53.0559882733517
    },
    "cts120970002": {
      "longitude": -81.4704,
      "latitude": 28.19627,
      "demand": 31.207623184084188
    }
  },
  "groceryStore": {
    "egs120970054": {
      "longitude": -81.43882,
      "latitude": 28.18148
    }
  }
}
"""

if __name__ == "__main__":
    # Test parsing
    parsed_data = parse_input_data(sample_data_json)
    print("Parsed Data:")
    print(json.dumps(parsed_data, indent=2))
    print("\n")

    # Extract grocery store and a sample stop for distance/time testing
    grocery_store_id = list(parsed_data["groceryStore"].keys())[0]
    grocery_store_coords = parsed_data["groceryStore"][grocery_store_id]
    
    sample_stop_id = list(parsed_data["transitStops"].keys())[0]
    sample_stop_coords = parsed_data["transitStops"][sample_stop_id]

    print(f"Grocery Store ({grocery_store_id}): Lat={grocery_store_coords['latitude']}, Lon={grocery_store_coords['longitude']}")
    print(f"Sample Stop ({sample_stop_id}): Lat={sample_stop_coords['latitude']}, Lon={sample_stop_coords['longitude']}, Demand={sample_stop_coords['demand']}")
    print("\n")

    # Test Haversine distance
    distance = haversine_distance(
        grocery_store_coords['latitude'], grocery_store_coords['longitude'],
        sample_stop_coords['latitude'], sample_stop_coords['longitude']
    )
    print(f"Distance between {grocery_store_id} and {sample_stop_id}: {distance:.2f} miles")

    # Test travel time
    travel_time = calculate_travel_time_minutes(distance)
    print(f"Estimated travel time: {travel_time:.2f} minutes")

    # Test with another stop
    if len(parsed_data["transitStops"]) > 1:
        another_stop_id = list(parsed_data["transitStops"].keys())[1]
        another_stop_coords = parsed_data["transitStops"][another_stop_id]
        distance_stops = haversine_distance(
            sample_stop_coords['latitude'], sample_stop_coords['longitude'],
            another_stop_coords['latitude'], another_stop_coords['longitude']
        )
        travel_time_stops = calculate_travel_time_minutes(distance_stops)
        print(f"\nDistance between {sample_stop_id} and {another_stop_id}: {distance_stops:.2f} miles")
        print(f"Estimated travel time between stops: {travel_time_stops:.2f} minutes")

