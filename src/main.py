import sys
import os
import csv
import traceback
import requests
import json
import logging
import argparse
import math
from sklearn.cluster import AgglomerativeClustering
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, request, jsonify, current_app
from src.utils.optimizer import run_simulation, recalibrate, GroceryStore, TransitStop, Bus, RoundTrip, RouteLeg
from src.utils.f2p_optimizer import run_f2p_simulation, recalibrate_f2p, GroceryStore as F2PGroceryStore, CommunityCenter, Truck, TruckTrip

app = Flask(__name__, template_folder="templates", static_folder="static")
app.logger.setLevel(logging.WARNING)

# Global error handler for Flask
@app.errorhandler(Exception)
def handle_global_exception(e):
    error_message = f"GLOBAL_FLASK_ERROR: An unhandled exception occurred: {str(e)}"
    current_app.logger.error(error_message)
    traceback.print_exc(file=sys.stderr)
    return jsonify(error="An internal server error occurred. Please check server logs."), 500

# Custom sys.excepthook for non-Flask uncaught exceptions
def custom_excepthook(exc_type, exc_value, exc_traceback):
    print("SYS_EXCEPTHOOK: Uncaught exception:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

sys.excepthook = custom_excepthook

MAPBOX_ACCESS_TOKEN = "***********************************************************" # Replace with you MAPBOX ACCESS TOKEN
MAX_INTERMEDIATE_STOPS_IN_CYCLE = 3
MAX_PICKUP_TIME_FROM_FS = 60

ALL_GROCERY_STORES = []
ALL_TRANSIT_STOPS = []
ALL_COMMUNITY_CENTERS = []

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GROCERY_STORES_CSV = os.path.join(DATA_DIR, "grocery_stores.csv")
TRANSIT_STOPS_CSV = os.path.join(DATA_DIR, "transit_stops.csv")
COMMUNITY_CENTERS_CSV = os.path.join(DATA_DIR, "community_centers.csv")

OUTER_RECALIBRATION_LOOPS = 80
INNER_RECALIBRATION_LOOPS = 8

MIN_TRANSIT_STOP_CLUSTERS = 3
MAX_TRANSIT_STOP_CLUSTERS = 0 # Set to 0 to disable clustering

def load_grocery_stores():
    global ALL_GROCERY_STORES
    ALL_GROCERY_STORES = []
    try:
        with open(GROCERY_STORES_CSV, mode="r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    store = {
                        "id": row.get("grocery store code") or row.get("id"),
                        "longitude": float(row.get("candid x (longitude)") or row.get("longitude")),
                        "latitude": float(row.get("candid y (latitude)") or row.get("latitude")),
                        "name": row.get("name", "N/A")
                    }
                    if not all([store["id"], store["longitude"] is not None, store["latitude"] is not None]):
                        current_app.logger.warning(f"Skipping grocery store row due to missing essential data: {row}")
                        continue
                    ALL_GROCERY_STORES.append(store)
                except (ValueError, TypeError) as e:
                    current_app.logger.error(f"Error processing grocery store row: {row}. Error: {e}")
        current_app.logger.info(f"Successfully loaded {len(ALL_GROCERY_STORES)} grocery stores from CSV.")
    except FileNotFoundError:
        current_app.logger.error(f"Grocery stores CSV file not found at {GROCERY_STORES_CSV}")
    except Exception as e:
        current_app.logger.error(f"An unexpected error occurred while loading grocery stores: {e}\n{traceback.format_exc()}")

def load_transit_stops():
    global ALL_TRANSIT_STOPS
    ALL_TRANSIT_STOPS = []
    try:
        with open(TRANSIT_STOPS_CSV, mode="r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    stop = {
                        "id": row.get("transit stop code") or row.get("id"),
                        "longitude": float(row.get("longitude")),
                        "latitude": float(row.get("latitude")),
                        "demand": int(float(row.get("demand")))
                    }
                    if not all([stop["id"], stop["longitude"] is not None, stop["latitude"] is not None, stop["demand"] is not None]):
                        current_app.logger.warning(f"Skipping transit stop row due to missing essential data: {row}")
                        continue
                    ALL_TRANSIT_STOPS.append(stop)
                except (ValueError, TypeError) as e:
                    current_app.logger.error(f"Error processing transit stop row: {row}. Error: {e}")
        current_app.logger.info(f"Successfully loaded {len(ALL_TRANSIT_STOPS)} transit stops from CSV.")
    except FileNotFoundError:
        current_app.logger.error(f"Transit stops CSV file not found at {TRANSIT_STOPS_CSV}")
    except Exception as e:
        current_app.logger.error(f"An unexpected error occurred while loading transit stops: {e}\n{traceback.format_exc()}")

def load_community_centers():
    global ALL_COMMUNITY_CENTERS
    ALL_COMMUNITY_CENTERS = []
    try:
        with open(COMMUNITY_CENTERS_CSV, mode="r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    center = {
                        "id": row.get("candid cc code"),
                        "longitude": float(row.get("longitude")),
                        "latitude": float(row.get("latitude")),
                        "demand": float(row.get("demand")),
                        "name": row.get("candid cc name", "N/A"),
                        "type": row.get("candid cc type", "N/A")
                    }
                    if not all([center["id"], center["longitude"] is not None, center["latitude"] is not None, center["demand"] is not None]):
                        current_app.logger.warning(f"Skipping community center row due to missing essential data: {row}")
                        continue
                    ALL_COMMUNITY_CENTERS.append(center)
                except (ValueError, TypeError) as e:
                    current_app.logger.error(f"Error processing community center row: {row}. Error: {e}")
        current_app.logger.info(f"Successfully loaded {len(ALL_COMMUNITY_CENTERS)} community centers from CSV.")
    except FileNotFoundError:
        current_app.logger.error(f"Community centers CSV file not found at {COMMUNITY_CENTERS_CSV}")
    except Exception as e:
        current_app.logger.error(f"An unexpected error occurred while loading community centers: {e}\n{traceback.format_exc()}")

with app.app_context():
    load_grocery_stores()
    load_transit_stops()
    load_community_centers()

def get_mapbox_matrix_durations(locations):
    """Fetches travel time matrix from Mapbox Matrix API."""
    if not locations or len(locations) < 2:
        return None, None, "Not enough locations for matrix API (min 2 required)."
    if len(locations) > 25:
        current_app.logger.warning(f"Attempting to get matrix for {len(locations)} locations, Mapbox limit is 25. This might fail or be slow.")

    coordinates_str = ";".join([f"{loc.longitude},{loc.latitude}" for loc in locations])
    matrix_url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coordinates_str}"
    params = {
        "access_token": MAPBOX_ACCESS_TOKEN,
        "annotations": "duration"
    }
    try:
        response = requests.get(matrix_url, params=params, timeout=60) 
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "Ok" and "durations" in data:
            location_to_idx_map = {loc.id: i for i, loc in enumerate(locations)}
            return data["durations"], location_to_idx_map, None
        else:
            error_message = data.get("message", "Unknown error from Matrix API")
            current_app.logger.error(f"Mapbox Matrix API error: {error_message} - Response: {data}")
            return None, None, error_message
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Mapbox Matrix API request failed: {e}")
        return None, None, str(e)
    except Exception as e:
        current_app.logger.error(f"Unexpected error in get_mapbox_matrix_durations: {e}\n{traceback.format_exc()}")
        return None, None, f"Unexpected error fetching matrix: {str(e)}"

def get_mapbox_directions_geometry(location_ids):
    """Fetches route geometry from Mapbox Directions API for a single leg."""
    if not location_ids or len(location_ids) < 2:
        return None, "Not enough locations for Directions API (min 2 required for a leg)."
    if len(location_ids) > 25:
        current_app.logger.warning(f"Attempting to get directions for {len(location_ids)} waypoints. Mapbox limit is 25.")

    locations = [location_dict[location_id] for location_id in location_ids]
    coordinates_str = ";".join([f"{loc.longitude},{loc.latitude}" for loc in locations])
    directions_url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{coordinates_str}"
    params = {
        "access_token": MAPBOX_ACCESS_TOKEN,
        "geometries": "geojson",
        "overview": "full"
    }
    try:
        response = requests.get(directions_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "Ok" and data.get("routes"):
            return data["routes"][0]["geometry"], None
        else:
            error_message = data.get("message", "Unknown error from Directions API")
            current_app.logger.error(f"Mapbox Directions API error: {error_message} - Response: {data}")
            return None, error_message
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Mapbox Directions API request failed: {e}")
        return None, str(e)
    except Exception as e:
        current_app.logger.error(f"Unexpected error in get_mapbox_directions_geometry: {e}\n{traceback.format_exc()}")
        return None, f"Unexpected error fetching directions: {str(e)}"

def serialize_optimizer_output( bus_objects_list, leg_route_geometries=None):
    num_buses = len(bus_objects_list)
    total_driving_time = sum(bus.total_weekly_driving_time for bus in bus_objects_list)
    total_passengers_serviced = sum(trip.pickup_leg.passengers_serviced for bus in bus_objects_list for day, trips in bus.daily_schedule.items() for trip in trips)

    current_app.logger.info(f"Starting serialization. Num buses: {num_buses}, Bus objects: {len(bus_objects_list) if bus_objects_list else 0}")
    try:
        serialized_bus_objects = []
        if bus_objects_list:
            for bus_idx, bus in enumerate(bus_objects_list):
                current_app.logger.debug(f"Serializing bus {bus.id}")
                bus_dict = {
                    "bus_id": bus.id, # Changed from "id" to "bus_id" to match frontend expectation
                    "capacity": bus.capacity,
                    "total_weekly_driving_time": bus.total_weekly_driving_time,
                    "daily_time_utilized": bus.daily_time_utilized,
                    "assigned_grocery_store_id": getattr(bus, 'assigned_grocery_store_id', None),
                    "daily_schedule": {}
                }
                if bus.daily_schedule:
                    for day, trips in bus.daily_schedule.items():
                        current_app.logger.debug(f"Serializing trips for bus {bus.id}, day {day}. Number of trips: {len(trips) if trips else 0}")
                        bus_dict["daily_schedule"][day] = []
                        if trips:
                            for trip_idx, trip in enumerate(trips):
                                trip_id_str = f"{bus.id}-day{day}-trip{trip_idx}" 
                                pickup_ordered_locations = [location_dict[loc_id] for loc_id in trip.pickup_leg.ordered_location_ids]
                                dropoff_ordered_locations = [location_dict[loc_id] for loc_id in trip.dropoff_leg.ordered_location_ids]
                                trip_dict = {
                                    "trip_id": trip_id_str,
                                    "passengers": trip.total_passengers, # Changed from "total_passengers"
                                    "duration_minutes": trip.duration, # Changed from "duration"
                                    "assigned_grocery_store_id": getattr(trip, 'assigned_grocery_store_id', bus_dict["assigned_grocery_store_id"]),
                                    "pickup_leg": {
                                        "travel_time": trip.pickup_leg.travel_time,
                                        "passengers_serviced": trip.pickup_leg.passengers_serviced,
                                        "demand_serviced_per_stop": trip.pickup_leg.passengers_served_per_stop,
                                        "ordered_stops_with_store": [{"id": s.id, "latitude": s.latitude, "longitude": s.longitude, "name": getattr(s, "name", "N/A")} for s in pickup_ordered_locations],
                                        "route_description": trip.pickup_leg.ordered_location_ids # Added for frontend
                                    },
                                    "dropoff_leg": { 
                                        "travel_time": trip.dropoff_leg.travel_time,
                                        "passengers_serviced": trip.dropoff_leg.passengers_serviced,
                                        "demand_serviced_per_stop": trip.dropoff_leg.passengers_served_per_stop,
                                        "ordered_stops_with_store": [{"id": s.id, "latitude": s.latitude, "longitude": s.longitude, "name": getattr(s, "name", "N/A")} for s in dropoff_ordered_locations],
                                        "route_description": trip.dropoff_leg.ordered_location_ids # Added for frontend
                                    }
                                }
                                
                                key_pickup = (bus.id, day, trip_idx, "pickup")
                                key_dropoff = (bus.id, day, trip_idx, "dropoff")
                                if leg_route_geometries and key_pickup in leg_route_geometries:
                                   trip_dict["pickup_leg"]["geometry"] = leg_route_geometries[key_pickup]
                                if leg_route_geometries and key_dropoff in leg_route_geometries:
                                   trip_dict["dropoff_leg"]["geometry"] = leg_route_geometries[key_dropoff]
                                
                                bus_dict["daily_schedule"][day].append(trip_dict)
                                current_app.logger.debug(f"Serialized trip: {trip_dict}")
                serialized_bus_objects.append(bus_dict)
        
        final_output = {
            "num_buses": num_buses,
            "total_driving_time": total_driving_time,
            "total_passengers_serviced": total_passengers_serviced,
            "bus_details": serialized_bus_objects
        }
        current_app.logger.info(f"Serialization complete. Final output structure keys: {list(final_output.keys())}")
        current_app.logger.debug(f"Full serialized output (first bus detail if exists): {json.dumps(final_output['bus_details'][0] if final_output.get('bus_details') else {}, indent=2, default=str)}")
        return final_output
    except Exception as e:
        current_app.logger.error(f"Error during serialization of optimizer output: {e}\n{traceback.format_exc()}")
        raise

def serialize_f2p_optimizer_output(truck_objects_list, leg_route_geometries=None):
    num_trucks = len(truck_objects_list)
    total_driving_time = sum(truck.total_weekly_driving_time for truck in truck_objects_list)
    total_cubic_feet_delivered = sum(trip.cubic_feet_delivered for truck in truck_objects_list for day, trips in truck.daily_schedule.items() for trip in trips)
    current_app.logger.info(f"Starting F2P serialization. Num trucks: {num_trucks}, Truck objects: {len(truck_objects_list) if truck_objects_list else 0}")
    try:
        serialized_truck_objects = []
        if truck_objects_list:
            for truck_idx, truck in enumerate(truck_objects_list):
                current_app.logger.debug(f"Serializing truck {truck.id}")
                truck_dict = {
                    "truck_id": truck.id,
                    "capacity": truck.capacity,
                    "total_weekly_time": truck.total_weekly_time,
                    "daily_time_utilized": truck.daily_time_utilized,
                    "assigned_grocery_store_id": getattr(truck.assigned_grocery_store, 'id', None),
                    "daily_schedule": {}
                }
                if truck.daily_schedule:
                    for day, trips in truck.daily_schedule.items():
                        current_app.logger.debug(f"Serializing trips for truck {truck.id}, day {day}. Number of trips: {len(trips) if trips else 0}")
                        truck_dict["daily_schedule"][day] = []
                        if trips:
                            for trip_idx, trip in enumerate(trips):
                                trip_id_str = f"{truck.id}-day{day}-trip{trip_idx}" 
                                trip_ordered_locations = [location_dict[loc_id] for loc_id in trip.ordered_location_ids]
                                trip_dict = {
                                    "trip_id": trip_id_str,
                                    "cubic_feet": trip.cubic_feet_delivered,
                                    "duration_minutes": trip.total_time,
                                    "assigned_grocery_store_id": getattr(truck.assigned_grocery_store, 'id', None),
                                    "ordered_locations": [{"id": loc.id, "latitude": loc.latitude, "longitude": loc.longitude, "name": getattr(loc, "name", "N/A")} for loc in trip_ordered_locations],
                                    "route_description": trip.ordered_location_ids # Added for frontend
                                }
                                key_leg = (truck.id, day, trip_idx)
                                if leg_route_geometries and key_leg in leg_route_geometries:
                                    trip_dict["geometry"] = leg_route_geometries[key_leg]
                                
                                truck_dict["daily_schedule"][day].append(trip_dict)
                                current_app.logger.debug(f"Serialized trip: {trip_dict}")
                serialized_truck_objects.append(truck_dict)
        
        final_output = {
            "num_trucks": num_trucks,
            "total_driving_time": total_driving_time,
            "total_cubic_feet_delivered": total_cubic_feet_delivered,
            "truck_details": serialized_truck_objects
        }
        current_app.logger.info(f"F2P Serialization complete. Final output structure keys: {list(final_output.keys())}")
        return final_output
    except Exception as e:
        current_app.logger.error(f"Error during serialization of F2P optimizer output: {e}\n{traceback.format_exc()}")
        raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/all_locations", methods=["GET"])
def get_all_locations():
    return jsonify({
        "grocery_stores": ALL_GROCERY_STORES,
        "transit_stops": ALL_TRANSIT_STOPS,
        "community_centers": ALL_COMMUNITY_CENTERS
    })


def cluster_transit_stops(initial_transit_stop_ids, mapbox_durations_matrix, location_to_idx_map=None):
    """
    Cluster transit stops into different numbers of clusters from 1 to min(10, number_of_stops).
    
    Args:
        initial_transit_stops (list): List of TransitStop objects
        
    Returns:
        dict: Dictionary where keys are number of clusters and values are dictionaries 
                mapping cluster IDs to lists of transit stop codes
    """

    # Dictionary to store results
    clustering_results = {
        0 : {label:[ts_id] for label, ts_id in enumerate(initial_transit_stop_ids)}
    }

    if MAX_TRANSIT_STOP_CLUSTERS == 0:
        return clustering_results
    
    if MIN_TRANSIT_STOP_CLUSTERS > MAX_TRANSIT_STOP_CLUSTERS:
        current_app.logger.warning(f"MIN_TRANSIT_STOP_CLUSTERS ({MIN_TRANSIT_STOP_CLUSTERS}) is greater than MAX_TRANSIT_STOP_CLUSTERS ({MAX_TRANSIT_STOP_CLUSTERS}). Clustering will not be performed.")
        return clustering_results

    transit_stop_idxs = [v for k, v in location_to_idx_map.items() if k in initial_transit_stop_ids]

    stop_durations_matrix = np.array(mapbox_durations_matrix)[np.ix_(transit_stop_idxs, transit_stop_idxs)]
    stop_durations_matrix = 0.5 * (stop_durations_matrix + stop_durations_matrix.T)  # Make it symmetric

    stop_to_idx_map = {stop_id: idx for idx, stop_id in enumerate(initial_transit_stop_ids)}
    idx_to_stop_map = {idx: stop_id for idx, stop_id in enumerate(initial_transit_stop_ids)}

    # Number of transit stops
    n_stops = len(initial_transit_stop_ids)
    
    if n_stops == 0:
        return {}
    
    # Maximum number of clusters to try
    max_clusters = min(MAX_TRANSIT_STOP_CLUSTERS, n_stops)

    # Minimum number of clusters to try
    min_clusters = max(MIN_TRANSIT_STOP_CLUSTERS, 1)
    
    # Perform clustering for different numbers of clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        # Initialize KMeans
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        
        # Fit the model
        cluster_labels = model.fit_predict(stop_durations_matrix)
        
        # Create the cluster assignment dictionary for this number of clusters
        cluster_assignments = {i: [] for i in range(n_clusters)}
        
        # Assign stops to clusters
        for i, label in enumerate(cluster_labels):
            stop_code = idx_to_stop_map[i]
            cluster_assignments[label].append(stop_code)
        
        # Store in the results dictionary
        clustering_results[n_clusters] = cluster_assignments
    
    return clustering_results


@app.route("/optimize", methods=["POST"])
def optimize_routes():
    current_app.logger.info("Received request to /optimize route")
    data = None
    try:
        current_app.logger.info(f"Attempting request.get_json()")
        data = request.get_json()
        current_app.logger.info(f"Raw data from request.get_json(): {data}")
        
        gs_data_list = data.get("selected_grocery_stores") 
        ts_data_list = data.get("selected_transit_stops")
        bus_capacity = data.get("bus_capacity", 15)
        current_app.logger.info(f"Parsed gs_data_list: {gs_data_list}, ts_data_list: {ts_data_list}, bus_capacity: {bus_capacity}")

        if not gs_data_list or not isinstance(gs_data_list, list) or not ts_data_list:
            current_app.logger.error(f"Missing or invalid gs_data_list or ts_data_list. gs_data_list: {gs_data_list}, ts_data_list: {ts_data_list}")
            return jsonify({"error": "Missing or invalid selected grocery stores (must be a list) or transit stops data for optimization"}), 400

        current_app.logger.info(f"Attempting to create GroceryStore and TransitStop objects.")
        initial_grocery_stores = [] 
        for gs_data in gs_data_list:
            initial_grocery_stores.append(
                GroceryStore(
                    id=gs_data["id"], 
                    latitude=float(gs_data["latitude"]),
                    longitude=float(gs_data["longitude"]),
                    name=gs_data.get("name", "N/A")
                )
            )
        
        initial_transit_stops = [
            TransitStop(id=ts["id"], latitude=float(ts["latitude"]), longitude=float(ts["longitude"]), total_demand=int(ts["demand"]))
            for ts in ts_data_list
        ]
        current_app.logger.info(f"Successfully created initial_grocery_stores: {initial_grocery_stores} and initial_transit_stops: {initial_transit_stops}")
        
        if not initial_transit_stops:
            current_app.logger.error(f"No transit stops selected for optimization. initial_transit_stops: {initial_transit_stops}")
            return jsonify({"error": "No transit stops selected for optimization."}), 400
        
        global location_dict
        location_dict = {loc.id: loc for loc in initial_grocery_stores + initial_transit_stops}


        all_selected_locations_for_matrix = initial_grocery_stores + initial_transit_stops
        unique_locations_dict = {loc.id: loc for loc in all_selected_locations_for_matrix}
        all_unique_locations_list = list(unique_locations_dict.values())

        mapbox_durations_matrix, location_to_idx_map, matrix_error = get_mapbox_matrix_durations(all_unique_locations_list)
        if matrix_error:
            current_app.logger.error(f"Error fetching Mapbox Matrix: {matrix_error}")
            return jsonify({"error": f"Error fetching travel time matrix: {matrix_error}"}), 500
        
        current_app.logger.info(f"Successfully fetched Mapbox Matrix. Matrix size: {len(mapbox_durations_matrix)}x{len(mapbox_durations_matrix[0]) if mapbox_durations_matrix else 0}")
        
        ts_id_list = [ts.id for ts in initial_transit_stops]
        transit_stop_clusters = cluster_transit_stops(ts_id_list, mapbox_durations_matrix, location_to_idx_map)
        current_app.logger.warning(f"clustered transit stops: \n{json.dumps(transit_stop_clusters, indent=2)}")

        num_buses, total_driving_time, schedule_data, bus_objects_list = run_simulation(
            initial_transit_stops=initial_transit_stops,
            initial_grocery_stores=initial_grocery_stores,
            mapbox_durations_matrix=mapbox_durations_matrix,
            location_to_idx_map=location_to_idx_map,
            bus_capacity=bus_capacity,
            logger=current_app.logger,
            max_intermediate_stops_in_cycle=MAX_INTERMEDIATE_STOPS_IN_CYCLE,
            max_pickup_time_from_fs=MAX_PICKUP_TIME_FROM_FS,
            outer_recalibration_loops= OUTER_RECALIBRATION_LOOPS,
            inner_recalibration_loops=INNER_RECALIBRATION_LOOPS,
            transit_stop_clusters=transit_stop_clusters
        )
        
        current_app.logger.info(f"Optimization complete. Num buses: {num_buses}, Total driving time: {total_driving_time}")
        
        if num_buses <= 0 or not bus_objects_list:
            current_app.logger.error(f"Optimization failed or returned no buses. num_buses: {num_buses}")
            return jsonify({"error": "Optimization failed or returned no valid bus schedules."}), 500
        
        leg_route_geometries = {}
        for bus in bus_objects_list:
            for day, trips in bus.daily_schedule.items():
                for trip_idx, trip in enumerate(trips):
                    pickup_leg_location_ids = trip.pickup_leg.ordered_location_ids
                    if len(pickup_leg_location_ids) >= 2:
                        pickup_geometry, pickup_error = get_mapbox_directions_geometry(pickup_leg_location_ids)
                        if pickup_geometry:
                            leg_route_geometries[(bus.id, day, trip_idx, "pickup")] = pickup_geometry
                        else:
                            current_app.logger.warning(f"Failed to get pickup leg geometry: {pickup_error}")
                    
                    dropoff_leg_location_ids = trip.dropoff_leg.ordered_location_ids
                    if len(dropoff_leg_location_ids) >= 2:
                        dropoff_geometry, dropoff_error = get_mapbox_directions_geometry(dropoff_leg_location_ids)
                        if dropoff_geometry:
                            leg_route_geometries[(bus.id, day, trip_idx, "dropoff")] = dropoff_geometry
                        else:
                            current_app.logger.warning(f"Failed to get dropoff leg geometry: {dropoff_error}")
        
        current_app.logger.info(f"Fetched {len(leg_route_geometries)} leg route geometries.")
        
        serialized_output = serialize_optimizer_output(bus_objects_list, leg_route_geometries)
        
        # Store optimization state for potential recalibration
        app.config['LAST_OPTIMIZATION_STATE'] = {
            'num_buses': num_buses,
            'total_driving_time': total_driving_time,
            'schedule_data': schedule_data,
            'bus_objects_list': bus_objects_list,
            'mapbox_durations_matrix': mapbox_durations_matrix,
            'location_to_idx_map': location_to_idx_map,
            'leg_route_geometries': leg_route_geometries,
            'initial_grocery_stores': initial_grocery_stores,
            'transit_stop_clusters': transit_stop_clusters
        }
        
        return jsonify(serialized_output)
    
    except Exception as e:
        current_app.logger.error(f"Error in optimize_routes: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An error occurred during optimization: {str(e)}"}), 500

@app.route("/recalibrate", methods=["POST"])
def recalibrate_routes():
    current_app.logger.info("Received request to /recalibrate route")
    
    try:
        # Check if we have a previous optimization state
        if 'LAST_OPTIMIZATION_STATE' not in app.config:
            current_app.logger.error("No previous optimization state found for recalibration")
            return jsonify({
                "error": "No previous optimization state found. Please run optimization first.",
                "recalibration_status": "failed",
                "recalibration_message": "No previous optimization to recalibrate."
            }), 400
        
        # Get the previous optimization state
        opt_state = app.config['LAST_OPTIMIZATION_STATE']
        num_buses = opt_state['num_buses']
        current_total_driving_time = opt_state['total_driving_time']
        current_schedule_data = opt_state['schedule_data']
        current_buses = opt_state['bus_objects_list']
        mapbox_durations_matrix = opt_state['mapbox_durations_matrix']
        location_to_idx_map = opt_state['location_to_idx_map']
        leg_route_geometries = opt_state.get('leg_route_geometries')
        initial_grocery_stores = opt_state['initial_grocery_stores']
        transit_stop_clusters = opt_state.get('transit_stop_clusters', {})

        
        current_app.logger.info(f"Starting recalibration with {num_buses} buses and current total driving time: {current_total_driving_time}")
        
        # Run recalibration
        new_bus_objects_list, recalibration_result = recalibrate(
            current_buses=current_buses,
            initial_grocery_stores=initial_grocery_stores,
            mapbox_durations_matrix=mapbox_durations_matrix,
            location_to_idx_map=location_to_idx_map,
            logger=current_app.logger,
            max_intermediate_stops_in_cycle=MAX_INTERMEDIATE_STOPS_IN_CYCLE,
            max_pickup_time_from_fs=MAX_PICKUP_TIME_FROM_FS,
            transit_stop_clusters=transit_stop_clusters
        )
        new_total_driving_time = sum(bus.total_weekly_driving_time for bus in new_bus_objects_list)
        new_num_buses = len(new_bus_objects_list)
        new_schedule_data = {bus.id: bus.daily_schedule for bus in new_bus_objects_list}
        
        current_app.logger.info(f"Recalibration complete. New total driving time: {new_total_driving_time}, Original: {current_total_driving_time}")
        
        # Determine recalibration status
        recalibration_status = "success"
        recalibration_message = ""
        
        if recalibration_result == 3:
            recalibration_status = "no_change"
            recalibration_message = "Recalibration found nothing to calibrate."
        elif recalibration_result == 2:
            recalibration_status = "no_improvement"
            recalibration_message = "Recalibration did not improve the solution."
        elif recalibration_result == 1:
            # Driving time decreased - improvement
            time_saved = current_total_driving_time - new_total_driving_time
            recalibration_status = "improved"
            recalibration_message = f"Recalibration improved the solution! Total driving time reduced by {time_saved:.2f} minutes."
        if recalibration_status == "improved":
            # Get route geometries for the new solution
            leg_route_geometries = {}
            for bus in new_bus_objects_list:
                for day, trips in bus.daily_schedule.items():
                    for trip_idx, trip in enumerate(trips):
                        pickup_leg_location_ids = trip.pickup_leg.ordered_location_ids
                        if len(pickup_leg_location_ids) >= 2:
                            pickup_geometry, pickup_error = get_mapbox_directions_geometry(pickup_leg_location_ids)
                            if pickup_geometry:
                                leg_route_geometries[(bus.id, day, trip_idx, "pickup")] = pickup_geometry
                            else:
                                current_app.logger.warning(f"Failed to get pickup leg geometry: {pickup_error}")
                        
                        dropoff_leg_location_ids = trip.pickup_leg.ordered_location_ids
                        if len(dropoff_leg_location_ids) >= 2:
                            dropoff_geometry, dropoff_error = get_mapbox_directions_geometry(dropoff_leg_location_ids)
                            if dropoff_geometry:
                                leg_route_geometries[(bus.id, day, trip_idx, "dropoff")] = dropoff_geometry
                            else:
                                current_app.logger.warning(f"Failed to get dropoff leg geometry: {dropoff_error}")
            
            current_app.logger.info(f"Fetched {len(leg_route_geometries)} leg route geometries for recalibrated solution.")
        else:
            # No need to fetch new route geometries if no improvement
            current_app.logger.info(f"Using existing leg route geometries for recalibrated solution. Count: {len(leg_route_geometries)}")
        
        # Serialize the output
        serialized_output = serialize_optimizer_output(new_bus_objects_list, leg_route_geometries)
        
        # Add recalibration status to the output
        serialized_output["recalibration_status"] = recalibration_status
        serialized_output["recalibration_message"] = recalibration_message
        serialized_output["previous_total_driving_time"] = current_total_driving_time
        
        # Update the optimization state for potential further recalibration
        app.config['LAST_OPTIMIZATION_STATE'] = {
            'num_buses': new_num_buses,
            'total_driving_time': new_total_driving_time,
            'schedule_data': new_schedule_data,
            'bus_objects_list': new_bus_objects_list,
            'mapbox_durations_matrix': mapbox_durations_matrix,
            'location_to_idx_map': location_to_idx_map,
            'leg_route_geometries': leg_route_geometries,
            'initial_grocery_stores': initial_grocery_stores,
            'transit_stop_clusters': transit_stop_clusters
        }
        
        return jsonify(serialized_output)
    
    except Exception as e:
        current_app.logger.error(f"Error in recalibrate_routes: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"An error occurred during recalibration: {str(e)}",
            "recalibration_status": "failed",
            "recalibration_message": f"Recalibration failed: {str(e)}"
        }), 500

@app.route("/optimize_f2p", methods=["POST"])
def optimize_f2p_routes():
    current_app.logger.info("Received request to /optimize_f2p route")
    data = None
    try:
        current_app.logger.info(f"Attempting request.get_json()")
        data = request.get_json()
        current_app.logger.info(f"Raw data from request.get_json(): {data}")
        
        gs_data_list = data.get("selected_grocery_stores") 
        cc_data_list = data.get("selected_community_centers")
        truck_capacity = data.get("truck_capacity", 270)
        current_app.logger.info(f"Parsed gs_data_list: {gs_data_list}, cc_data_list: {cc_data_list}, truck_capacity: {truck_capacity}")

        if not gs_data_list or not isinstance(gs_data_list, list) or not cc_data_list:
            current_app.logger.error(f"Missing or invalid gs_data_list or cc_data_list. gs_data_list: {gs_data_list}, cc_data_list: {cc_data_list}")
            return jsonify({"error": "Missing or invalid selected grocery stores (must be a list) or community centers data for optimization"}), 400

        current_app.logger.info(f"Attempting to create GroceryStore and CommunityCenter objects.")
        initial_grocery_stores = [] 
        for gs_data in gs_data_list:
            initial_grocery_stores.append(
                F2PGroceryStore(
                    id=gs_data["id"], 
                    latitude=float(gs_data["latitude"]),
                    longitude=float(gs_data["longitude"]),
                    name=gs_data.get("name", "N/A")
                )
            )
        
        initial_community_centers = [
            CommunityCenter(
                id=cc["id"], 
                latitude=float(cc["latitude"]), 
                longitude=float(cc["longitude"]), 
                demand=float(cc["demand"]),
                name=cc.get("name", "N/A")
            )
            for cc in cc_data_list
        ]
        current_app.logger.info(f"Successfully created initial_grocery_stores: {initial_grocery_stores} and initial_community_centers: {initial_community_centers}")
        
        if not initial_community_centers:
            current_app.logger.error(f"No community centers selected for optimization. initial_community_centers: {initial_community_centers}")
            return jsonify({"error": "No community centers selected for optimization."}), 400
        
        initial_cc_id_to_cc_dict= {cc.id: cc for cc in initial_community_centers}

        global location_dict
        location_dict = {loc.id: loc for loc in initial_grocery_stores + initial_community_centers}

        all_selected_locations_for_matrix = initial_grocery_stores + initial_community_centers
        unique_locations_dict = {loc.id: loc for loc in all_selected_locations_for_matrix}
        all_unique_locations_list = list(unique_locations_dict.values())
        
        current_app.logger.info(f"Fetching Mapbox matrix durations for {len(all_unique_locations_list)} unique locations.")
        mapbox_durations_matrix, location_to_idx_map, matrix_error = get_mapbox_matrix_durations(all_unique_locations_list)
        if matrix_error:
            current_app.logger.error(f"Error fetching Mapbox matrix durations: {matrix_error}")
            return jsonify({"error": f"Error fetching travel time matrix: {matrix_error}"}), 500
        
        current_app.logger.info(f"Running F2P simulation with {len(initial_grocery_stores)} grocery stores, {len(initial_community_centers)} community centers, and truck capacity {truck_capacity}.")
        num_trucks, total_driving_time, truck_objects_list = run_f2p_simulation(
            initial_grocery_stores, 
            initial_cc_id_to_cc_dict, 
            truck_capacity, 
            mapbox_durations_matrix, 
            location_to_idx_map,
            logger=current_app.logger,
            outer_recalibration_loops= OUTER_RECALIBRATION_LOOPS,
            inner_recalibration_loops=INNER_RECALIBRATION_LOOPS
        )
        
        current_app.logger.info(f"F2P Simulation complete. Num trucks: {num_trucks}, Total driving time: {total_driving_time}")
        
        # Get route geometries for all trips
        leg_route_geometries = {}
        for truck in truck_objects_list:
            for day, trips in truck.daily_schedule.items():
                for trip_idx, trip in enumerate(trips):
                    location_ids = trip.ordered_location_ids
                    if len(location_ids) >= 2:
                        geometry, error = get_mapbox_directions_geometry(location_ids)
                        if geometry:
                            leg_route_geometries[(truck.id, day, trip_idx)] = geometry
                        else:
                            current_app.logger.warning(f"Failed to get trip geometry: {error}")
        
        current_app.logger.info(f"Fetched {len(leg_route_geometries)} leg route geometries.")
        
        serialized_output = serialize_f2p_optimizer_output(truck_objects_list, leg_route_geometries)
        
        app.config['LAST_F2P_OPTIMIZATION_STATE'] = {
            'num_trucks': num_trucks,
            'total_driving_time': total_driving_time,
            'truck_objects_list': truck_objects_list,
            'mapbox_durations_matrix': mapbox_durations_matrix,
            'location_to_idx_map': location_to_idx_map,
            'leg_route_geometries': leg_route_geometries,
            'truck_capacity': truck_capacity,
            'initial_grocery_stores': initial_grocery_stores
        }
        
        return jsonify(serialized_output)
    
    except Exception as e:
        current_app.logger.error(f"Error in optimize_f2p_routes: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An error occurred during F2P optimization: {str(e)}"}), 500

@app.route("/recalibrate_f2p", methods=["POST"])
def recalibrate_f2p_routes():
    current_app.logger.info("Received request to /recalibrate_f2p route")
    try:
        if 'LAST_F2P_OPTIMIZATION_STATE' not in app.config:
            current_app.logger.error("No previous F2P optimization state found for recalibration.")
            return jsonify({"error": "No previous F2P optimization to recalibrate. Please run F2P optimization first."}), 400
        
        last_state = app.config['LAST_F2P_OPTIMIZATION_STATE']
        current_app.logger.info(f"Retrieved last F2P optimization state with {len(last_state['truck_objects_list'])} trucks.")
        
        current_num_trucks = last_state['num_trucks']
        current_total_driving_time = last_state['total_driving_time']
        current_truck_objects_list = last_state['truck_objects_list']
        mapbox_durations_matrix = last_state['mapbox_durations_matrix']
        location_to_idx_map = last_state['location_to_idx_map']
        leg_route_geometries = last_state['leg_route_geometries']
        truck_capacity = last_state['truck_capacity']
        initial_grocery_stores = last_state['initial_grocery_stores']
        
        current_app.logger.info(f"Running F2P recalibration with {current_num_trucks} trucks and {current_total_driving_time} total driving time.")
        new_truck_objects_list, recalibration_result = recalibrate_f2p(
            current_truck_objects_list,
            initial_grocery_stores,
            mapbox_durations_matrix,
            location_to_idx_map,
            logger=current_app.logger
        )

        new_num_trucks = len(new_truck_objects_list)
        new_total_driving_time = sum(truck.total_weekly_driving_time for truck in new_truck_objects_list)
        
        current_app.logger.info(f"F2P Recalibration complete. Result: {recalibration_result}, New num trucks: {new_num_trucks}, New total driving time: {new_total_driving_time}")
        
        # Determine recalibration status
        recalibration_status = "success"
        recalibration_message = ""
        
        if recalibration_result == 3:
            recalibration_status = "no_change"
            recalibration_message = "Recalibration found nothing to calibrate."
        elif recalibration_result == 2:
            recalibration_status = "no_improvement"
            recalibration_message = "Recalibration did not improve the solution."
        elif recalibration_result == 1:
            # Driving time decreased - improvement
            time_saved = current_total_driving_time - new_total_driving_time
            recalibration_status = "improved"
            recalibration_message = f"Recalibration improved the solution! Total driving time reduced by {time_saved:.2f} minutes."
        
        if recalibration_status == "improved":
            # Get route geometries for the new solution
            leg_route_geometries = {}
            for truck in new_truck_objects_list:
                for day, trips in truck.daily_schedule.items():
                    for trip_idx, trip in enumerate(trips):
                        location_ids = trip.ordered_location_ids
                        if len(location_ids) >= 2:
                            geometry, error = get_mapbox_directions_geometry(location_ids)
                            if geometry:
                                leg_route_geometries[(truck.id, day, trip_idx)] = geometry
                            else:
                                current_app.logger.warning(f"Failed to get trip geometry: {error}")
            
            current_app.logger.info(f"Fetched {len(leg_route_geometries)} leg route geometries for recalibrated F2P solution.")
        else:
            # No need to fetch new route geometries if no improvement
            current_app.logger.info(f"Using existing leg route geometries for recalibrated F2P solution. Count: {len(leg_route_geometries)}")
        
        # Serialize the output
        serialized_output = serialize_f2p_optimizer_output(new_truck_objects_list, leg_route_geometries)
        
        # Add recalibration status to the output
        serialized_output["recalibration_status"] = recalibration_status
        serialized_output["recalibration_message"] = recalibration_message
        serialized_output["previous_total_driving_time"] = current_total_driving_time
        
        # Update the optimization state for potential further recalibration
        app.config['LAST_F2P_OPTIMIZATION_STATE'] = {
            'num_trucks': new_num_trucks,
            'total_driving_time': new_total_driving_time,
            'truck_objects_list': new_truck_objects_list,
            'mapbox_durations_matrix': mapbox_durations_matrix,
            'location_to_idx_map': location_to_idx_map,
            'leg_route_geometries': leg_route_geometries,
            'truck_capacity': truck_capacity,
            'initial_grocery_stores': initial_grocery_stores
        }
        
        return jsonify(serialized_output)
    
    except Exception as e:
        current_app.logger.error(f"Error in recalibrate_f2p_routes: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"An error occurred during F2P recalibration: {str(e)}",
            "recalibration_status": "failed",
            "recalibration_message": f"F2P Recalibration failed: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Route Planner Flask Application')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()
    
    # Get port from environment variable if available, otherwise use command line argument
    port = int(os.environ.get('PORT', args.port))
    host = os.environ.get('HOST', args.host)
    
    print(f"Starting server on {host}:{port}")
    app.run(debug=True, host=host, port=port)
