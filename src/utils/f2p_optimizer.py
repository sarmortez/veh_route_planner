import math
import itertools
import json
import copy
import sys
import logging
import random
from collections import deque
from typing import Dict, List, Any, Union

LOGGER = logging.getLogger(__name__)

# Constants for F2P operations
MAX_DAILY_OPERATION_HOURS = 8  # 8 hours per day
GROCERY_STORE_LOADING_TIME = 45  # 45 minutes loading time at grocery store
COMMUNITY_CENTER_UNLOADING_TIME = 15  # 15 minutes unloading time per visit
CUBIC_FEET_PER_HOUSEHOLD = 18  # 1 household needs 18 cubic feet of food per week
LONG_TRAVEL_TIME_THRESHOLD_TRUCK = 60 # 60 minutes is considered a long travel time
MEDIUM_TRAVEL_TIME_THRESHOLD_TRUCK = 20 # 20 minutes is considered a medium travel time
UNDER_UTILIZED_DAILY_OPERATION_HOURS = 4  # 4 hours is considered under-utilized daily operation hours
# Set up logging

# --- Data Structures ---
class Location:
    def __init__(self, id, latitude, longitude):
        self.id = id
        self.latitude = latitude
        self.longitude = longitude

class CommunityCenter(Location):
    def __init__(self, id, latitude, longitude, demand, name="N/A"):
        super().__init__(id, latitude, longitude)
        # Round up the demand as per requirements
        self.demand = math.ceil(demand)
        self.name = name
        # Calculate cubic feet needed (1 demand unit = 1 household = 18 cubic feet)
        self.cubic_feet_needed = self.demand * CUBIC_FEET_PER_HOUSEHOLD
        self.cubic_feet_remaining = self.cubic_feet_needed

    def add_delivery(self, cubic_feet_delivered):
        """Record a delivery to this community center"""
        self.cubic_feet_remaining = max(0, self.cubic_feet_remaining - cubic_feet_delivered)
        
    def remove_delivery(self, cubic_feet_removed):
        """Remove a delivery (used when recalibrating)"""
        self.cubic_feet_remaining = min(self.cubic_feet_needed, self.cubic_feet_remaining + cubic_feet_removed)
        
    def is_fully_served(self):
        """Check if this community center's demand is fully met"""
        return self.cubic_feet_remaining <= 0
    
    def __repr__(self):
        return f"CommunityCenter(ID: {self.id}, Demand: {self.demand}, Remaining: {self.cubic_feet_remaining} cubic ft)"

class GroceryStore(Location):
    def __init__(self, id, latitude, longitude, name="N/A"):
        super().__init__(id, latitude, longitude)
        self.name = name
        
    def __repr__(self):
        return f"GroceryStore(ID: {self.id}, Name: {self.name})"

class TruckTrip:
    def __init__(self, ordered_location_ids, travel_time_minutes, cubic_feet_delivered, cubic_feet_per_center):
        self.ordered_location_ids = ordered_location_ids  # List of location IDs in order of visit
        # ordered_locations = [location_dict.get(loc_id, None) for loc_id in ordered_location_ids]  # List of locations in order of visit
        self.travel_time = travel_time_minutes  # Total travel time for this leg
        self.cubic_feet_delivered = cubic_feet_delivered  # Total cubic feet delivered in this leg
        self.cubic_feet_per_center = cubic_feet_per_center  # Dict mapping center ID to cubic feet delivered
        
        # Calculate total operational time including loading/unloading
        self.total_time = travel_time_minutes + GROCERY_STORE_LOADING_TIME
        # Add unloading time for each community center
        for loc_id in ordered_location_ids:
            if cubic_feet_per_center.get(loc_id, 0) > 0:
                self.total_time += COMMUNITY_CENTER_UNLOADING_TIME
                
    def __repr__(self):
        return f"TruckTrip(Route: {' -> '.join(self.ordered_location_ids)}, Time: {self.travel_time:.2f}min, Delivered: {self.cubic_feet_delivered} cubic ft)"

# class TruckTrip:
#     def __init__(self, grocery_store, delivery_legs):
#         self.grocery_store = grocery_store
#         self.delivery_legs = delivery_legs  # List of TruckDeliveryLeg objects
        
#         # Calculate total cubic feet delivered
#         self.total_cubic_feet_delivered = sum(leg.cubic_feet_delivered for leg in delivery_legs)
        
#         # Calculate total operational time including loading at grocery store
#         self.total_time = sum(leg.total_time for leg in delivery_legs)
#         # Add loading time at grocery store for each leg
#         self.total_time += len(delivery_legs) * GROCERY_STORE_LOADING_TIME
        
#     def __repr__(self):
#         return f"TruckTrip(Store: {self.grocery_store.id}, Legs: {len(self.delivery_legs)}, Time: {self.total_time:.2f}min, Delivered: {self.total_cubic_feet_delivered} cubic ft)"

class Truck:
    def __init__(self, id, capacity=270):
        self.id = id
        self.capacity = capacity  # Cubic feet capacity
        self.daily_schedule = {day: [] for day in range(5)}  # Monday to Friday
        self.daily_time_utilized = {day: 0.0 for day in range(5)}
        self.daily_driving_time = {day: 0.0 for day in range(5)}  # Total driving time per day
        self.daily_cubic_feet_delivered = {day: 0 for day in range(5)}
        self.total_weekly_time = 0.0
        self.total_weekly_cubic_feet = 0
        self.total_weekly_driving_time = 0.0
        self.assigned_grocery_store = None
        self.days_to_recalibrate = []
        
    def add_trip(self, day, truck_trip):
        """Add a trip to this truck's schedule for the specified day"""
        if day not in self.daily_schedule:
            raise ValueError(f"Invalid day: {day}. Must be 0-4 (Monday-Friday)")
            
        # Check if adding this trip would exceed daily operation hours
        new_daily_time = self.daily_time_utilized[day] + truck_trip.total_time
        if new_daily_time > (MAX_DAILY_OPERATION_HOURS * 60):
            return False
            
        self.daily_schedule[day].append(truck_trip)
        self.daily_time_utilized[day] += truck_trip.total_time
        self.daily_cubic_feet_delivered[day] += truck_trip.cubic_feet_delivered
        self.daily_driving_time[day] += truck_trip.travel_time
        self.total_weekly_time += truck_trip.total_time
        self.total_weekly_cubic_feet += truck_trip.cubic_feet_delivered
        self.total_weekly_driving_time += truck_trip.travel_time
        return True
    
    def mark_days_to_recalibrate(self):
        for day in range(5):
            if (
                self.daily_time_utilized[day] > (MAX_DAILY_OPERATION_HOURS * 60)
                or self.daily_time_utilized[day] < (UNDER_UTILIZED_DAILY_OPERATION_HOURS * 60)
                or any(
                    trip.travel_time > LONG_TRAVEL_TIME_THRESHOLD_TRUCK
                    for trip in self.daily_schedule[day]
                )
            ):
                self.days_to_recalibrate.append(day)
        self.days_to_recalibrate = list(set(self.days_to_recalibrate))
        
    def reset_day_for_recalibration(self, day, reference_cc_dict="main"):
        """Reset a specific day's schedule"""
        if day not in self.days_to_recalibrate:
            LOGGER.warning(f"Day {day} is not marked for recalibration. Cannot reset.")
            return {}
        
        if reference_cc_dict == "temp":
            cc_id_to_cc_dict = temp_cc_id_to_cc_dict
        elif reference_cc_dict == "main":
            cc_id_to_cc_dict = main_cc_id_to_cc_dict
        else:
            raise ValueError(f"Invalid reference_cc_dict: {reference_cc_dict}. Must be 'temp' or 'main'.")
            
        # Get centers that need to have their deliveries restored
        center_ids_to_recalibrate = set()
        for trip in self.daily_schedule[day]:
            for loc_id in trip.ordered_location_ids:
                cubic_feet_to_remove = trip.cubic_feet_per_center.get(loc_id, 0)
                if cubic_feet_to_remove > 0:
                    loc = cc_id_to_cc_dict.get(loc_id)
                    loc.remove_delivery(cubic_feet_to_remove)
                    center_ids_to_recalibrate.add(loc_id)
                        
        # Update truck stats
        self.total_weekly_time -= self.daily_time_utilized[day]
        self.total_weekly_cubic_feet -= self.daily_cubic_feet_delivered[day]
        self.total_weekly_driving_time -= self.daily_driving_time[day]
        self.daily_schedule[day] = []
        self.daily_time_utilized[day] = 0.0
        self.daily_cubic_feet_delivered[day] = 0
        self.daily_driving_time[day] = 0.0

        self.days_to_recalibrate.remove(day)        

        return center_ids_to_recalibrate
        
    def reset_for_new_week(self):
        """Reset the truck for a new week of planning"""
        self.daily_schedule = {day: [] for day in range(5)}
        self.daily_time_utilized = {day: 0.0 for day in range(5)}
        self.daily_cubic_feet_delivered = {day: 0 for day in range(5)}
        self.daily_driving_time = {day: 0.0 for day in range(5)}  # Total driving time per day
        self.total_weekly_time = 0.0
        self.total_weekly_cubic_feet = 0
        self.assigned_grocery_store = None
        self.total_weekly_driving_time = 0.0
        
    def __repr__(self):
        return f"Truck(ID: {self.id}, Capacity: {self.capacity} cubic ft, Weekly Time: {self.total_weekly_time:.2f}min, Weekly Delivered: {self.total_weekly_cubic_feet} cubic ft)"


def random_weave(data: Dict[Union[str, int], List[Any]]) -> List[Any]:
    """
    Return ONE order-preserving interleaving of the value-lists in `data`,
    chosen uniformly at random from the full set of possibilities.

    Proof sketch of uniformity
    --------------------------
    •  Let the remaining lengths of the k queues be r₁, …, r_k (Σ r_i = N).
    •  The number of valid interleavings that start by taking an element
       from queue *j* is: (N-1)! / ∏_{i≠j} r_i! / (r_j−1)!  =  r_j × (N-1)! / ∏ r_i!
    •  Therefore the probability that a uniformly random weave begins with
       queue *j* is  r_j / N.
    •  Re-applying the argument after every draw keeps the distribution
       uniform all the way to the end.
    """
    # turn each list into a queue for O(1) pops from the left
    queues = [deque(lst) for lst in data.values()]
    remaining = [len(q) for q in queues]   # parallel list of lengths

    result = []
    total_left = sum(remaining)

    while total_left:
        # choose a queue index with probability proportional to its remaining length
        idx = random.choices(range(len(queues)), weights=remaining, k=1)[0]

        # pull the next item from that queue
        result.append(queues[idx].popleft())

        # bookkeeping
        remaining[idx] -= 1
        total_left -= 1
        if remaining[idx] == 0:            # queue exhausted → drop it
            queues.pop(idx)
            remaining.pop(idx)

    return result


def get_matrix_travel_time_minutes(loc1_id, loc2_id, matrix_durations, location_to_idx_map):
    """Get travel time between two locations from the matrix"""
    try:
        idx1 = location_to_idx_map[loc1_id]
        idx2 = location_to_idx_map[loc2_id]
        duration_seconds = matrix_durations[idx1][idx2]
        return duration_seconds / 60.0
    except KeyError as e:
        LOGGER.error(f"Error: Location ID {e} not found in location_to_idx_map.")
        return float("inf")
    except IndexError:
        LOGGER.error(f"Error: Matrix index out of bounds for {loc1_id} or {loc2_id}.")
        return float("inf")

# def calculate_tsp_route(start_point, intermediate_stops, end_point, matrix_durations, location_to_idx_map):
#     """Calculate the optimal route using TSP algorithm"""
#     if not intermediate_stops:
#         time_minutes = get_matrix_travel_time_minutes(start_point.id, end_point.id, matrix_durations, location_to_idx_map)
#         return ([start_point, end_point], time_minutes)

#     min_total_time_minutes = float("inf")
#     best_path_locations = []
    
#     # For small number of stops, try all permutations
#     if len(intermediate_stops) <= 8:
#         for perm_tuple in itertools.permutations(intermediate_stops):
#             perm = list(perm_tuple)
#             current_total_time_minutes = 0
#             current_path_locations = [start_point] + perm + [end_point]
            
#             # Calculate total time for this permutation
#             current_total_time_minutes += get_matrix_travel_time_minutes(start_point.id, perm[0].id, matrix_durations, location_to_idx_map)
#             for i in range(len(perm) - 1):
#                 current_total_time_minutes += get_matrix_travel_time_minutes(perm[i].id, perm[i+1].id, matrix_durations, location_to_idx_map)
#             current_total_time_minutes += get_matrix_travel_time_minutes(perm[-1].id, end_point.id, matrix_durations, location_to_idx_map)
            
#             if current_total_time_minutes < min_total_time_minutes:
#                 min_total_time_minutes = current_total_time_minutes
#                 best_path_locations = current_path_locations
#     else:
#         # For larger sets, use a greedy approach
#         current_location = start_point
#         current_path = [current_location]
#         remaining_stops = intermediate_stops.copy()
        
#         while remaining_stops:
#             next_stop = min(remaining_stops, 
#                            key=lambda stop: get_matrix_travel_time_minutes(current_location.id, stop.id, matrix_durations, location_to_idx_map))
#             current_path.append(next_stop)
#             current_location = next_stop
#             remaining_stops.remove(next_stop)
            
#         current_path.append(end_point)
#         best_path_locations = current_path
        
#         # Calculate total time for this path
#         min_total_time_minutes = 0
#         for i in range(len(best_path_locations) - 1):
#             min_total_time_minutes += get_matrix_travel_time_minutes(
#                 best_path_locations[i].id, best_path_locations[i+1].id, matrix_durations, location_to_idx_map)
    
#     if not best_path_locations:
#         return (None, float("inf"))
        
#     return (best_path_locations, min_total_time_minutes)

def calculate_tsp_route_mapbox(start_point, intermediate_stops, end_point, matrix_durations, location_to_idx_map):
    if not intermediate_stops:
        time_minutes = get_matrix_travel_time_minutes(start_point.id, end_point.id, matrix_durations, location_to_idx_map)
        return ([start_point, end_point], time_minutes)

    min_total_time_minutes = float("inf")
    best_path_locations = []
    for perm_tuple in itertools.permutations(intermediate_stops):
        perm = list(perm_tuple)
        current_total_time_minutes = 0
        current_path_locations = [start_point] + perm + [end_point]
        current_total_time_minutes += get_matrix_travel_time_minutes(start_point.id, perm[0].id, matrix_durations, location_to_idx_map)
        for i in range(len(perm) - 1):
            current_total_time_minutes += get_matrix_travel_time_minutes(perm[i].id, perm[i+1].id, matrix_durations, location_to_idx_map)
        current_total_time_minutes += get_matrix_travel_time_minutes(perm[-1].id, end_point.id, matrix_durations, location_to_idx_map)
        if current_total_time_minutes < min_total_time_minutes:
            min_total_time_minutes = current_total_time_minutes
            best_path_locations = current_path_locations
    if not best_path_locations:
        return (None, float("inf"))
    return (best_path_locations, min_total_time_minutes)


def calculate_optimized_cycle_components_trucks(grocery_store, cycle_intermediate_centers, matrix_durations, location_to_idx_map):
    if not cycle_intermediate_centers:
        return None
    components = {
        # "cycle_intermediate_centers": cycle_intermediate_centers
    }
    tsp_path, tsp_time = calculate_tsp_route_mapbox(grocery_store, cycle_intermediate_centers, grocery_store, matrix_durations, location_to_idx_map)
    if not tsp_path or tsp_time == float("inf"):
        return None
    # components["tsp_path"] = tsp_path
    components["tsp_path_ids"] = [loc.id for loc in tsp_path]
    components["tsp_time"] = tsp_time
    return components


def find_best_truck_trip(
        grocery_store, 
        cc_id_list, 
        truck_capacity, 
        matrix_durations, 
        location_to_idx_map, 
        remaining_minutes,
        max_intermediate_centers_in_cycle=3,
        logger=LOGGER,
        reference_cc_dict="main",
        allow_trips='long'):
    """Plan a single delivery leg from grocery store to community centers and back"""
    # if not community_centers:
    #     return None
        
    # Sort centers by remaining demand (descending)
    # sorted_centers = sorted(community_centers, key=lambda cc: cc.cubic_feet_remaining, reverse=True)
    if reference_cc_dict == "temp":
        cc_id_to_cc_dict = temp_cc_id_to_cc_dict
    elif reference_cc_dict == "main":
        cc_id_to_cc_dict = main_cc_id_to_cc_dict
    else:
        raise ValueError(f"Invalid reference_cc_dict: {reference_cc_dict}. Must be 'temp' or 'main'.")

    community_centers = [cc_id_to_cc_dict[cc_id] for cc_id in cc_id_list]

    active_centers = [cc for cc in community_centers if cc.cubic_feet_remaining > 0]
    if not active_centers:
        return None
    
    total_cubic_feet_remaining = sum(cc.cubic_feet_remaining for cc in active_centers)

    if total_cubic_feet_remaining < truck_capacity * 23:
        max_intermediate_centers_in_cycle += 1 # Allow one more stop in the cycle if demand is low (as we are towards the end of the scheduling process)
    elif total_cubic_feet_remaining < truck_capacity * 15:
        max_intermediate_centers_in_cycle += 2 # Allow two more stops in the cycle if demand is low (as we are towards the end of the scheduling process)
    elif total_cubic_feet_remaining < truck_capacity * 8:
        max_intermediate_centers_in_cycle += 3 # Allow three more stops in the cycle if demand is low (as we are towards the end of the scheduling process)
    elif total_cubic_feet_remaining < truck_capacity * 3:
        max_intermediate_centers_in_cycle += 4 # Allow four more stops in the cycle if demand is low (as we are towards the end of the scheduling process)
        # max_intermediate_stops_in_cycle = len(active_stops) # Allow all stops to be used in the cycle when demand is low (towards the end of the scheduling process)

    best_metric_so_far_val = (0, -float("inf"))
    best_cycle_info = None

    for num_centers_in_cycle_combo in range(1, min(len(active_centers), max_intermediate_centers_in_cycle) + 1):
        for current_center_combination_tuple in itertools.combinations(active_centers, num_centers_in_cycle_combo):
            current_center_combination_list = list(current_center_combination_tuple)
            
            cycle_components = calculate_optimized_cycle_components_trucks(grocery_store, current_center_combination_list, matrix_durations, location_to_idx_map)

            if not cycle_components:
                continue

            tsp_time = cycle_components["tsp_time"]
            
            cycle_time = tsp_time + GROCERY_STORE_LOADING_TIME + COMMUNITY_CENTER_UNLOADING_TIME * len(current_center_combination_list)

            if cycle_time > remaining_minutes:
                # If the cycle time exceeds remaining minutes, skip this combination
                continue

            cycle_cubic_feet_remaining = sum(cc.cubic_feet_remaining for cc in current_center_combination_list)

            cycle_cubic_feet_delivered = min(cycle_cubic_feet_remaining, truck_capacity)

            cycle_metric = (cycle_cubic_feet_delivered, - cycle_time)

            if cycle_metric > best_metric_so_far_val:
                best_metric_so_far_val = cycle_metric

                # Update the best cycle components
                best_cycle_info = cycle_components
                best_cycle_info["cubic_feet_delivered"] = cycle_cubic_feet_delivered
                best_cycle_info["cycle_time"] = cycle_time
    

    if best_cycle_info is None:
        logger.info("No valid cycle found for the given parameters.")
        return None
    
    if allow_trips == 'medium' and best_cycle_info['tsp_time'] > LONG_TRAVEL_TIME_THRESHOLD_TRUCK:
        return None
    if allow_trips == 'short' and best_cycle_info['tsp_time'] > MEDIUM_TRAVEL_TIME_THRESHOLD_TRUCK:
        return None
    
    logger.info(f"Best Cycle Components: {best_cycle_info}")

    load_to_deliver_this_cycle = best_cycle_info["cubic_feet_delivered"]
    cubic_feet_per_center = {}

    for center_id in best_cycle_info["tsp_path_ids"][1:-1]:  # Skip grocery store at start and end
        center = cc_id_to_cc_dict.get(center_id)

        if center.cubic_feet_remaining <= 0:
            continue
            
        # Calculate how much we can deliver to this center
        delivery_amount = min(center.cubic_feet_remaining, load_to_deliver_this_cycle)

        if delivery_amount > 0:
            center.add_delivery(delivery_amount)
            cubic_feet_per_center[center.id] = delivery_amount

            load_to_deliver_this_cycle -= delivery_amount
            
            # If truck is full, stop adding centers
            if load_to_deliver_this_cycle <= 0:
                break

    best_cycle_info["cubic_feet_per_center"] = cubic_feet_per_center

    truck_trip = TruckTrip(
        ordered_location_ids=best_cycle_info['tsp_path_ids'],
        travel_time_minutes=best_cycle_info['tsp_time'],
        cubic_feet_delivered=best_cycle_info['cubic_feet_delivered'],
        cubic_feet_per_center=best_cycle_info["cubic_feet_per_center"]
    )

    return truck_trip


def plan_truck_daily_trips(
        grocery_store, 
        cc_id_list, 
        truck_capacity, 
        matrix_durations, 
        location_to_idx_map,
        logger=LOGGER,
        reference_cc_dict="main",
        allow_trips='long'): # options are 'long', 'medium', and 'short'
    """Plan all trips for a truck for a single day"""
    daily_trips = []
    day_remaining_minutes = MAX_DAILY_OPERATION_HOURS * 60

    if reference_cc_dict == "temp":
        cc_id_to_cc_dict = temp_cc_id_to_cc_dict
    elif reference_cc_dict == "main":
        cc_id_to_cc_dict = main_cc_id_to_cc_dict
    else:
        raise ValueError(f"Invalid reference_cc_dict: {reference_cc_dict}. Must be 'temp' or 'main'.")
    
    # Continue planning trips until we run out of time or all centers are served
    while day_remaining_minutes > 0:
        # Check if all centers are fully served
        community_centers = [cc_id_to_cc_dict[cc_id] for cc_id in cc_id_list]

        if all(center.is_fully_served() for center in community_centers):
            break
            
        # Plan a delivery leg
        truck_trip = find_best_truck_trip(
            grocery_store, 
            cc_id_list, 
            truck_capacity, 
            matrix_durations, 
            location_to_idx_map, 
            day_remaining_minutes,
            logger=logger,
            reference_cc_dict=reference_cc_dict,
            allow_trips=allow_trips,
        )
        
        if not truck_trip:
            break
            
        # Calculate time needed for this leg including loading time
        trip_time = truck_trip.total_time
            
        daily_trips.append(truck_trip)
        
        # Update remaining time
        day_remaining_minutes -= trip_time
    
    return daily_trips

def run_f2p_simulation(
        grocery_stores, 
        cc_id_to_cc_dict, 
        truck_capacity, 
        matrix_durations, 
        location_to_idx_map,
        logger=LOGGER,
        outer_recalibration_loops=50,
        inner_recalibration_loops=10,
        max_intermediate_stops_in_cycle=3):
    """Run the F2P optimization simulation"""
    logger.info(f"Starting F2P simulation with {len(grocery_stores)} grocery stores, {len(cc_id_to_cc_dict)} community centers")
    
    # Make copies of the community centers to avoid modifying the originals

    global main_cc_id_to_cc_dict, temp_cc_id_to_cc_dict
    main_cc_id_to_cc_dict = cc_id_to_cc_dict
    cc_id_list = list(main_cc_id_to_cc_dict.keys())
    
    # Calculate total demand across all community centers
    total_cubic_feet_needed = sum(cc.cubic_feet_needed for cc in main_cc_id_to_cc_dict.values())
    remaining_cubic_feet = total_cubic_feet_needed
    logger.info(f"Total demand: {total_cubic_feet_needed} cubic feet")
    
    # Initialize trucks
    trucks = []
    truck_id = 1
    best_cc_id_to_cc_dict = main_cc_id_to_cc_dict

    while remaining_cubic_feet > 0:

        best_remaining_cubic_feet = sum(cc.cubic_feet_remaining for cc in best_cc_id_to_cc_dict.values())
        best_truck_total_weekly_driving_time = float("inf")
        best_truck_total_weekly_cubic_feet = -float("inf")
        best_metric = (best_remaining_cubic_feet, best_truck_total_weekly_driving_time, -best_truck_total_weekly_cubic_feet)
        
        for grocery_store in grocery_stores:
            temp_cc_id_to_cc_dict = copy.deepcopy(main_cc_id_to_cc_dict)
            # temp_community_centers = copy.deepcopy(community_centers_copy)

            temp_truck = Truck(f"T{truck_id}", truck_capacity)
            logger.info(f"Working on Truck {temp_truck.id}...")
            logger.info(f"Assigning Truck {temp_truck.id} to Grocery Store {grocery_store.id}")
            temp_truck.assigned_grocery_store = grocery_store

            for day in range(5):
                logger.info(f"Planning day {day+1}")
                # Plan daily trips for this truck
                daily_trips = plan_truck_daily_trips(
                    grocery_store, 
                    cc_id_list, 
                    truck_capacity, 
                    matrix_durations, 
                    location_to_idx_map,
                    logger=logger,
                    reference_cc_dict="temp")

                for truck_trip in daily_trips:
                    # Add trip to truck's schedule
                    if not temp_truck.add_trip(day, truck_trip):
                        logger.warning(f"Could not add trip to Truck {temp_truck.id} on day {day}. Trip exceeds daily time limit.")

            # Calculate remaining cubic feet after initial planning
            temp_community_centers = [temp_cc_id_to_cc_dict[cc_id] for cc_id in cc_id_list]
            temp_remaining_cubic_feet = sum(cc.cubic_feet_remaining for cc in temp_community_centers)

            # Calculate total weekly driving time for this truck
            temp_truck_total_weekly_driving_time = temp_truck.total_weekly_driving_time

            # Calculate total cubic feet delivered by this truck
            temp_truck_total_weekly_cubic_feet = temp_truck.total_weekly_cubic_feet

            temp_metric = (temp_remaining_cubic_feet, temp_truck_total_weekly_driving_time, - temp_truck_total_weekly_cubic_feet)

            if temp_metric < best_metric:
                logger.info(f"New best grocery store found: {grocery_store.id}")
                best_metric = temp_metric
                best_truck = temp_truck
                best_cc_id_to_cc_dict = temp_cc_id_to_cc_dict
        
        if remaining_cubic_feet == best_metric[0]:
            logger.warning(f"No truck can cover the remaining demand given the current constraints. remaining_cubic_feet: {remaining_cubic_feet}")
            return -1, 0, []

        trucks.append(best_truck)
        remaining_cubic_feet = best_metric[0]
        main_cc_id_to_cc_dict = best_cc_id_to_cc_dict
        truck_id += 1

    logger.info(f"Initial planning complete. Total trucks needed: {len(trucks)}")

    if not trucks:
        logger.warning("No trucks were created during the initial planning phase.")
        return -1, 0, []


    best_trucks = copy.deepcopy(trucks)
    best_truck_total_weekly_driving_time = sum(truck.total_weekly_driving_time for truck in best_trucks)

    for i in range(outer_recalibration_loops):
        temp_trucks = copy.deepcopy(best_trucks)
        for j in range(inner_recalibration_loops):
            (
                temp_trucks,
                recalibration_status
            ) = recalibrate_f2p(
                temp_trucks,
                grocery_stores,
                matrix_durations=matrix_durations, 
                location_to_idx_map=location_to_idx_map,
                logger=logger
            )

            if recalibration_status == 1:
                temp_truck_total_weekly_driving_time = sum(truck.total_weekly_driving_time for truck in temp_trucks)
                if temp_truck_total_weekly_driving_time < best_truck_total_weekly_driving_time:
                    logger.info(f"Found better truck configuration in inner loop {j+1} of outer loop {i+1}.")
                    best_trucks = copy.deepcopy(temp_trucks)
                    best_truck_total_weekly_driving_time = temp_truck_total_weekly_driving_time

    total_weekly_time = sum(truck.total_weekly_time for truck in best_trucks)
    return len(best_trucks), total_weekly_time, best_trucks
    

def reset_days_to_recalibrate_f2p(truck_objects_list, reference_cc_dict="main"):
    truck_days_to_recalibrate = {}
    center_ids_to_recalibrate = set()
    for truck in truck_objects_list:
        truck.mark_days_to_recalibrate()
        days_to_recalibrate = sorted(set([(truck.id, day) for day in truck.days_to_recalibrate]), key=lambda x: x[1])
        truck_days_to_recalibrate[truck.id] = days_to_recalibrate
        for _, day in days_to_recalibrate:
            center_ids_to_recalibrate |= (truck.reset_day_for_recalibration(day, reference_cc_dict=reference_cc_dict))
        if len(days_to_recalibrate) == 5:
            truck.reset_for_new_week()
    
    return truck_days_to_recalibrate, list(center_ids_to_recalibrate)


def get_community_center_ids_from_trucks(trucks):
    """Extract community centers from the trucks' schedules"""
    community_center_ids = set()
    # Iterate through all trucks and their daily schedules
    for truck in trucks:
        for day, trips in truck.daily_schedule.items():
            for trip in trips:
                for loc_ids in trip.ordered_location_ids:
                    if loc_ids in trip.cubic_feet_per_center:
                        community_center_ids.add(loc_ids)
    return community_center_ids



def recalibrate_f2p(
        current_trucks,
        grocery_stores,
        matrix_durations, 
        location_to_idx_map,
        logger=LOGGER):
    """Attempt to recalibrate the F2P solution to improve efficiency"""
    logger.info(f"Starting F2P recalibration with {len(current_trucks)} trucks")
    
    if any(truck.assigned_grocery_store is None for truck in current_trucks):
        logger.warning("Some trucks do not have an assigned grocery store. Cannot recalibrate.")
    else:
        logger.info("All trucks have an assigned grocery store. Proceeding with recalibration.")

    current_total_weekly_driving_time = sum(truck.total_weekly_driving_time for truck in current_trucks)

    # Make copies to avoid modifying originals
    new_trucks = copy.deepcopy(current_trucks)
    truck_id_to_truck = {truck.id: truck for truck in new_trucks}
    new_center_ids = get_community_center_ids_from_trucks(new_trucks)
    
    global temp_cc_id_to_cc_dict, main_cc_id_to_cc_dict
    temp_cc_id_to_cc_dict = copy.deepcopy(main_cc_id_to_cc_dict)

    new_centers = [temp_cc_id_to_cc_dict[cc_id] for cc_id in new_center_ids]

    cubic_feet_covered_by_trucks = sum(trip.cubic_feet_delivered for truck in new_trucks for day, trips in truck.daily_schedule.items() for trip in trips)
    total_cubic_feet_of_all_centers = sum(cc.cubic_feet_needed for cc in new_centers)
    remaining_cubic_feet_all = sum(cc.cubic_feet_remaining for cc in new_centers)
    cubic_feet_covered_at_all_centers = total_cubic_feet_of_all_centers - remaining_cubic_feet_all
    if cubic_feet_covered_by_trucks != cubic_feet_covered_at_all_centers:
        logger.warning(f"Before reseting days to recalibrate, cubic_feet_covered_by_trucks: {cubic_feet_covered_by_trucks} does not match cubic_feet_covered_at_all_centers: {cubic_feet_covered_at_all_centers}. This is unexpected.")

    
    trucks_days_to_recalibrate, center_ids_to_recalibrate = reset_days_to_recalibrate_f2p(new_trucks, reference_cc_dict="temp")


    if len(center_ids_to_recalibrate) == 0:
        logger.info("No trucks to recalibrate. Returning original trucks and community centers.")
        recalibration_status = 3  # No recalibration needed
        return current_trucks, recalibration_status

    centers_to_recalibrate = [temp_cc_id_to_cc_dict[cc_id] for cc_id in center_ids_to_recalibrate]
    remaining_cubic_feet = sum(cc.cubic_feet_remaining for cc in centers_to_recalibrate)
    logger.info(f"remaining_cubic_feet in the beginning before recalibration: {remaining_cubic_feet}")
    remaining_cubic_feet_all = sum(cc.cubic_feet_remaining for cc in new_centers)
    if remaining_cubic_feet != remaining_cubic_feet_all:
        logger.warning(f"After reseting days to recalibrate, remaining cubic feet in centers to recalibrate: {remaining_cubic_feet} does not match remaining cubic feet in all centers: {remaining_cubic_feet_all}. This is unexpected.")
    
    total_cubic_feet_of_all_centers = sum(cc.cubic_feet_needed for cc in new_centers)
    logger.info(f"total_cubic_feet_of_all_centers: {total_cubic_feet_of_all_centers}")
    total_cubic_feet_of_centers_to_recalibrate = sum(cc.cubic_feet_needed for cc in centers_to_recalibrate)
    logger.info(f"total_cubic_feet_of_centers_to_recalibrate: {total_cubic_feet_of_centers_to_recalibrate}")

    cubic_feet_covered_at_all_centers = total_cubic_feet_of_all_centers - remaining_cubic_feet_all
    cubic_feet_covered_by_trucks = sum(trip.cubic_feet_delivered for truck in new_trucks for day, trips in truck.daily_schedule.items() for trip in trips)
    
    if cubic_feet_covered_by_trucks != cubic_feet_covered_at_all_centers:
        logger.warning(f"After reseting days to recalibrate, cubic_feet_covered_by_trucks: {cubic_feet_covered_by_trucks} does not match cubic_feet_covered_at_all_centers: {cubic_feet_covered_at_all_centers}. This is unexpected.")


    random_schedule_order = random_weave(trucks_days_to_recalibrate)

    new_total_weekly_driving_time = 0.0
    for truck_id, day in random_schedule_order:

        logger.info(f"Recalibrating Truck {truck_id} for day {day+1}")

        truck = truck_id_to_truck[truck_id]

        truck_grocery_store = truck.assigned_grocery_store

        if truck_grocery_store is None:
            # assiign a random grocery store to the truck
            logger.info(f"Truck {truck_id} assigned grocery store is being reset.")
            truck_grocery_store = random.choice(grocery_stores)
            truck.assigned_grocery_store = truck_grocery_store
        
        logger.info(f"Truck {truck_id} assigned to Grocery Store {truck.assigned_grocery_store.id}")

        allow_trips = random.choice(['long', 'medium', 'short'])  # Randomly choose trip type for recalibration
        # allow_trips = 'long'  # Force long trips for recalibration 

        best_truck_daily_trips = plan_truck_daily_trips(
            grocery_store=truck_grocery_store,
            cc_id_list=center_ids_to_recalibrate,
            truck_capacity=truck.capacity,
            matrix_durations=matrix_durations,
            location_to_idx_map=location_to_idx_map,
            logger=logger,
            reference_cc_dict="temp",
            allow_trips=allow_trips  # Allow trips based on the random choice 
        )

        if not best_truck_daily_trips:
            logger.info(f"Did not find a best daily trip schedule for Truck {truck_id} on day {day+1}.")
            continue
        
        logger.info(f"Found {len(best_truck_daily_trips)} daily trips for Truck {truck_id} on day {day+1}.")

        for truck_trip in best_truck_daily_trips:
            # Add trip to truck's schedule
            if truck.add_trip(day, truck_trip):
                logger.info(f"Added trip to Truck {truck.id} on day {day+1}: {truck_trip}")
            else:
                logger.warning(f"Could not add trip to Truck {truck.id} on day {day+1}. Trip exceeds daily time limit.")
    
    
    centers_to_recalibrate = [temp_cc_id_to_cc_dict[cc_id] for cc_id in center_ids_to_recalibrate]            
    remaining_cubic_feet = sum(cc.cubic_feet_remaining for cc in centers_to_recalibrate)

    new_total_weekly_driving_time = sum(truck.total_weekly_driving_time for truck in new_trucks)

    if new_total_weekly_driving_time >= current_total_weekly_driving_time or remaining_cubic_feet > 0:
        logger.info(f"Recalibration did not improve solution. Current time: {current_total_weekly_driving_time:.2f}, New time: {new_total_weekly_driving_time:.2f}")
        recalibration_status = 2
        return current_trucks, recalibration_status
    
    logger.info("Recalibration improved solution.")
    logger.info(f"Recalibration reduced total weekly driving time from {current_total_weekly_driving_time:.2f} to {new_total_weekly_driving_time:.2f} minutes")
    recalibration_status = 1  # Recalibration improved the solution

    g2truck_list = {}
    update_truck_list = False
    for truck in new_trucks:
        gs = truck.assigned_grocery_store
        if gs is None:
            logger.warning(f"Truck {truck.id} has no assigned grocery store. Skipping recalibration for this truck. Why?")
            continue
        truck_capacity = truck.capacity
        n_empty_days = len([day for day in truck.daily_schedule if not truck.daily_schedule[day]])
        if (gs.id, truck_capacity) not in g2truck_list:
            g2truck_list[(gs.id, truck_capacity)] = []
        g2truck_list[(gs.id, truck_capacity)].append((truck, n_empty_days))
    for (gs_id, truck_capacity), truck_list in g2truck_list.items():
        total_empty_days = sum(n_empty_days for _, n_empty_days in truck_list)
        while total_empty_days > 4:
            truck_list.sort(key=lambda x: x[1], reverse=False) # Sort by number of empty days
            truck_to_drop, n_empty_days = truck_list.pop() # Drop the truck with the most empty days
            logger.info(f"Truck {truck_to_drop.id} is being dropped from Grocery Store {gs_id} as its service can be covered by other trucks of the same capacity.")
            update_truck_list = True
            other_empty_truck_days = []
            for truck, n_empty_days in truck_list:
                for day in range(5-n_empty_days, 5):
                    other_empty_truck_days.append((truck.id, day))

            daily_schedule = truck_to_drop.daily_schedule
            for day in range(4, 0, -1):
                if len(daily_schedule[day]) > 0:
                    other_truck_id, day_to_reassign = other_empty_truck_days.pop(0)
                    other_truck = truck_id_to_truck[other_truck_id]
                    for trip in daily_schedule[day]:
                        other_truck.add_trip(day_to_reassign, trip)

            total_empty_days = sum(n_empty_days for _, n_empty_days in truck_list)
    
    if update_truck_list:
        new_trucks = [truck for gs_id, truck_list in g2truck_list.items() for truck, _ in truck_list]
        new_trucks.sort(key=lambda x: x.id)  # Sort trucks by ID for consistency
    
    recalibration_status = 1  # Recalibration improved the solution
    return new_trucks, recalibration_status