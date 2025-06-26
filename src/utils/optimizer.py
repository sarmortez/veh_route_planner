import math
import itertools
import json
import copy
import sys
import logging
import random
from collections import deque
from typing import Dict, List, Any, Union


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# Constants
MAX_DAILY_OPERATION_HOURS = 8
UNDER_UTILIZED_DAILY_OPERATION_HOURS = 4
LONG_TRAVEL_TIME_THRESHOLD = 20
STORE_WAIT_TIME_MINUTES = 40
STOP_WAIT_TIME_MINUTES = 2


# --- Data Structures ---
class Location:
    def __init__(self, id, latitude, longitude):
        self.id = id
        self.latitude = latitude
        self.longitude = longitude

class TransitStop(Location):
    def __init__(self, id, latitude, longitude, total_demand):
        super().__init__(id, latitude, longitude)
        self.total_demand = total_demand
        self.weekly_demand_remaining = total_demand

    def add_service(self, passengers_serviced):
        self.weekly_demand_remaining = max(0, self.weekly_demand_remaining - passengers_serviced)

    def remove_service(self, passengers_unserviced):
        self.weekly_demand_remaining = min(self.total_demand, self.weekly_demand_remaining + passengers_unserviced)

class GroceryStore(Location):
    def __init__(self, id, latitude, longitude, name="N/A"):
        super().__init__(id, latitude, longitude)
        self.name = name

class RouteLeg:
    def __init__(self, ordered_location_ids, travel_time_minutes, passengers_serviced, passengers_served_per_stop):
        self.ordered_location_ids = ordered_location_ids
        self.travel_time = travel_time_minutes
        self.passengers_serviced = passengers_serviced
        self.passengers_served_per_stop = passengers_served_per_stop

        self.total_time = travel_time_minutes
        for loc_id in ordered_location_ids:
            if passengers_served_per_stop.get(loc_id, -1) >= 0:
                self.total_time += STOP_WAIT_TIME_MINUTES

    def __repr__(self):
        stop_ids = [s.id for s in self.ordered_stops_with_store]
        return f"RouteLeg(Stops: {' -> '.join(stop_ids)}, Time: {self.travel_time:.2f}min, Pax: {self.passengers_serviced})"

class RoundTrip:
    def __init__(self, pickup_leg, dropoff_leg):
        self.pickup_leg = pickup_leg
        self.dropoff_leg = dropoff_leg
        self.total_passengers = pickup_leg.passengers_serviced
        self.duration = pickup_leg.total_time + STORE_WAIT_TIME_MINUTES + dropoff_leg.total_time

    def __repr__(self):
        return f"RoundTrip(Duration: {self.duration:.2f}min, Pax: {self.total_passengers}, Pickup: {self.pickup_leg}, Dropoff: {self.dropoff_leg})"

class Bus:
    def __init__(self, id, capacity=15):
        self.id = id
        self.capacity = capacity
        self.daily_schedule = {day: [] for day in range(5)}
        self.daily_time_utilized = {day: 0.0 for day in range(5)}
        self.daily_driving_time = {day: 0.0 for day in range(5)}
        self.daily_passengers_serviced = {day: 0 for day in range(5)}
        self.total_weekly_time_utilized = 0.0
        self.total_weekly_passengers_served = 0
        self.total_weekly_driving_time = 0.0
        self.assigned_grocery_store_id = None
        self.assigned_grocery_store = None
        self.days_to_recalibrate = []

    def add_trip(self, day, round_trip):
        if day not in self.daily_schedule:
            raise ValueError(f"Invalid day: {day}. Must be between 0 (Monday) and 4 (Friday).")
        
        new_daily_time_utilized = self.daily_time_utilized[day] + round_trip.duration
        if new_daily_time_utilized > (MAX_DAILY_OPERATION_HOURS * 60):
            return False
        self.daily_schedule[day].append(round_trip)
        self.daily_time_utilized[day] = new_daily_time_utilized
        self.daily_passengers_serviced[day] += round_trip.total_passengers
        self.daily_driving_time[day] += (round_trip.pickup_leg.travel_time + round_trip.dropoff_leg.travel_time)
        self.total_weekly_time_utilized += round_trip.duration
        self.total_weekly_driving_time += (round_trip.pickup_leg.travel_time + round_trip.dropoff_leg.travel_time)
        self.total_weekly_passengers_served += round_trip.total_passengers
        return True

    def mark_days_to_recalibrate(self):
        for day in range(5):
            if (
                self.daily_time_utilized[day] > (MAX_DAILY_OPERATION_HOURS * 60)
                or self.daily_time_utilized[day] < (UNDER_UTILIZED_DAILY_OPERATION_HOURS * 60)
                or self.daily_schedule[day][-1].pickup_leg.total_time > LONG_TRAVEL_TIME_THRESHOLD
            ):
                self.days_to_recalibrate.append(day)
        self.days_to_recalibrate = list(set(self.days_to_recalibrate))

    def reset_day_for_recalibration(self, day, reference_ts_dict="main"):
        if reference_ts_dict == "temp":
            ts_id_to_ts_dict = temp_ts_id_to_ts_dict
        elif reference_ts_dict == "main":
            ts_id_to_ts_dict = main_ts_id_to_ts_dict
        else:
            raise ValueError("Invalid reference_ts_dict value. Use 'temp' or 'main'.")

        stop_ids_to_recalibrate = set()

        if day in self.days_to_recalibrate:
            for round_trip in self.daily_schedule.get(day, []):
                pickup_leg = round_trip.pickup_leg
                ordered_location_ids = pickup_leg.ordered_location_ids
                passengers_served_per_stop = pickup_leg.passengers_served_per_stop
                for loc_id in ordered_location_ids:
                    passengers_served_to_remove = passengers_served_per_stop.get(loc_id, 0)
                    if passengers_served_to_remove > 0:
                        ts = ts_id_to_ts_dict.get(loc_id)
                        ts.remove_service(passengers_served_to_remove)
                        stop_ids_to_recalibrate.add(loc_id)
            
            # update bus stats
            self.total_weekly_time_utilized -= self.daily_time_utilized[day]
            self.total_weekly_driving_time -= self.daily_driving_time[day]
            self.total_weekly_passengers_served -= self.daily_passengers_serviced[day]
            self.daily_schedule[day] = []
            self.daily_time_utilized[day] = 0.0
            self.daily_driving_time[day] = 0.0
            self.daily_passengers_serviced[day] = 0

            self.days_to_recalibrate.remove(day)
            self.days_to_recalibrate = list(set(self.days_to_recalibrate))

            return stop_ids_to_recalibrate

        else:
            LOGGER.warning(f"Day {day} is not marked for recalibration.")

            return {}

    def reset_for_new_week(self):
        self.daily_schedule = {day: [] for day in range(5)}
        self.daily_time_utilized = {day: 0.0 for day in range(5)}
        self.daily_driving_time = {day: 0.0 for day in range(5)}
        self.total_weekly_time_utilized = 0.0
        self.total_weekly_driving_time = 0.0
        self.days_to_recalibrate = []
        self.assigned_grocery_store_id = None
        self.assigned_grocery_store = None

    def get_assigned_grocery_store(self):
        return self.assigned_grocery_store
    
    def __repr__(self):
        return f"Bus(ID: {self.id}, WeeklyDrivingTime: {self.total_weekly_driving_time:.2f}min)"
    
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
    try:
        idx1 = location_to_idx_map[loc1_id]
        idx2 = location_to_idx_map[loc2_id]
        duration_seconds = matrix_durations[idx1][idx2]
        return duration_seconds / 60.0
    except KeyError as e:
        print(f"Error: Location ID {e} not found in location_to_idx_map.")
        return float("inf")
    except IndexError:
        print(f"Error: Matrix index out of bounds for {loc1_id} or {loc2_id}.")
        return float("inf")

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

# def calculate_passenger_friendly_leg_details(grocery_store, cycle_intermediate_stops, matrix_durations, location_to_idx_map, leg_type):
#     if not cycle_intermediate_stops:
#         return None, float("inf")
#     farthest_cycle_stop = None
#     max_time_from_gs = -1
#     for stop in cycle_intermediate_stops:
#         time_to_stop = get_matrix_travel_time_minutes(grocery_store.id, stop.id, matrix_durations, location_to_idx_map)
#         if time_to_stop > max_time_from_gs:
#             max_time_from_gs = time_to_stop
#             farthest_cycle_stop = stop
#         elif time_to_stop == max_time_from_gs:
#             if farthest_cycle_stop is None or (stop.id < farthest_cycle_stop.id):
#                  farthest_cycle_stop = stop
#     if farthest_cycle_stop is None:
#         return None, float("inf")
#     ordered_leg_stops_for_routeleg = []
#     leg_total_time_minutes = 0
#     tsp_intermediate_anchored = [s for s in cycle_intermediate_stops if s.id != farthest_cycle_stop.id]
#     if leg_type == "pickup":
#         time_gs_to_farthest = get_matrix_travel_time_minutes(grocery_store.id, farthest_cycle_stop.id, matrix_durations, location_to_idx_map)
#         if time_gs_to_farthest == float("inf"):
#             return None, float("inf")
#         tsp_pickup_sequence, tsp_pickup_time = calculate_tsp_route_mapbox(farthest_cycle_stop, tsp_intermediate_anchored, grocery_store, matrix_durations, location_to_idx_map)
#         if not tsp_pickup_sequence or tsp_pickup_time == float("inf"):
#             return None, float("inf")
#         ordered_leg_stops_for_routeleg = [grocery_store] + tsp_pickup_sequence
#         leg_total_time_minutes = time_gs_to_farthest + tsp_pickup_time
#     elif leg_type == "dropoff":
#         time_farthest_to_gs = get_matrix_travel_time_minutes(farthest_cycle_stop.id, grocery_store.id, matrix_durations, location_to_idx_map)
#         if time_farthest_to_gs == float("inf"):
#             return None, float("inf")
#         tsp_dropoff_sequence, tsp_dropoff_time = calculate_tsp_route_mapbox(grocery_store, tsp_intermediate_anchored, farthest_cycle_stop, matrix_durations, location_to_idx_map)
#         if not tsp_dropoff_sequence or tsp_dropoff_time == float("inf"):
#             return None, float("inf")
#         ordered_leg_stops_for_routeleg = tsp_dropoff_sequence + [grocery_store]
#         leg_total_time_minutes = tsp_dropoff_time + time_farthest_to_gs
#     else:
#         return None, float("inf")
#     final_path = []
#     if ordered_leg_stops_for_routeleg:
#         final_path.append(ordered_leg_stops_for_routeleg[0])
#         for i in range(1, len(ordered_leg_stops_for_routeleg)):
#             if ordered_leg_stops_for_routeleg[i].id != ordered_leg_stops_for_routeleg[i-1].id:
#                 final_path.append(ordered_leg_stops_for_routeleg[i])
#     return final_path, leg_total_time_minutes

def calculate_optimized_cycle_components(grocery_store, cycle_intermediate_stops, matrix_durations, location_to_idx_map):
    if not cycle_intermediate_stops:
        return None
    farthest_cycle_stop = None
    max_time_from_gs_to_fs = -1
    for stop in cycle_intermediate_stops:
        time_to_stop = get_matrix_travel_time_minutes(grocery_store.id, stop.id, matrix_durations, location_to_idx_map)
        if time_to_stop == float("inf"):
            return None
        if time_to_stop > max_time_from_gs_to_fs:
            max_time_from_gs_to_fs = time_to_stop
            farthest_cycle_stop = stop
        elif time_to_stop == max_time_from_gs_to_fs:
            if farthest_cycle_stop is None or (stop.id < farthest_cycle_stop.id):
                 farthest_cycle_stop = stop
    if farthest_cycle_stop is None:
        return None
    components = {
        "farthest_stop_obj": farthest_cycle_stop,
        "cycle_intermediate_stops": cycle_intermediate_stops
    }
    components["time_gs_to_fs"] = get_matrix_travel_time_minutes(grocery_store.id, farthest_cycle_stop.id, matrix_durations, location_to_idx_map)
    components["path_gs_to_fs_list"] = [grocery_store, farthest_cycle_stop]
    if components["time_gs_to_fs"] == float("inf"):
        return None
    components["time_fs_to_gs"] = get_matrix_travel_time_minutes(farthest_cycle_stop.id, grocery_store.id, matrix_durations, location_to_idx_map)
    components["path_fs_to_gs_list"] = [farthest_cycle_stop, grocery_store]
    if components["time_fs_to_gs"] == float("inf"):
        return None
    tsp_intermediate_for_pickup = [s for s in cycle_intermediate_stops if s.id != farthest_cycle_stop.id]
    pickup_tsp_path, pickup_tsp_time = calculate_tsp_route_mapbox(farthest_cycle_stop, tsp_intermediate_for_pickup, grocery_store, matrix_durations, location_to_idx_map)
    if not pickup_tsp_path or pickup_tsp_time == float("inf"):
        return None
    components["tsp_pickup_path_from_fs_list"] = pickup_tsp_path
    components["tsp_pickup_time_from_fs"] = pickup_tsp_time
    tsp_intermediate_for_dropoff = [s for s in cycle_intermediate_stops if s.id != farthest_cycle_stop.id]
    dropoff_tsp_path, dropoff_tsp_time = calculate_tsp_route_mapbox(grocery_store, tsp_intermediate_for_dropoff, farthest_cycle_stop, matrix_durations, location_to_idx_map)
    if not dropoff_tsp_path or dropoff_tsp_time == float("inf"):
        return None
    components["tsp_dropoff_path_to_fs_list"] = dropoff_tsp_path
    components["tsp_dropoff_time_to_fs"] = dropoff_tsp_time
    return components

def find_best_daily_cycle(
        grocery_store, 
        ts_id_list,
        bus_capacity, 
        matrix_durations, 
        location_to_idx_map,
        max_intermediate_stops_in_cycle=3,
        logger=LOGGER, 
        reference_ts_dict="main",
        max_pickup_time_from_fs=60):
    
    if reference_ts_dict == "temp":
        ts_id_to_ts_dict = temp_ts_id_to_ts_dict
    elif reference_ts_dict == "main":
        ts_id_to_ts_dict = main_ts_id_to_ts_dict
    else:
        raise ValueError("Invalid reference_ts_dict value. Use 'temp' or 'main'.")

    transit_stop_list = [[ts_id_to_ts_dict[ts_id] for ts_id in ts_ids] for ts_ids in ts_id_list]
    # transit_stops = [ts_id_to_ts_dict[ts_id] for ts_id in ts_id_list]

    active_stops = [[stop for stop in transit_stops if stop.weekly_demand_remaining > 0] for transit_stops in transit_stop_list]
    active_stops = [stop_list for stop_list in active_stops if stop_list]  # Filter out empty lists
    if not active_stops:
        return None
    
    total_demand_remaining = sum(stop.weekly_demand_remaining for stop_list in active_stops for stop in stop_list)


    if total_demand_remaining < bus_capacity * 23:
        max_intermediate_stops_in_cycle += 1 # Allow one more stop in the cycle if demand is low (as we are towards the end of the scheduling process)
    elif total_demand_remaining < bus_capacity * 15:
        max_intermediate_stops_in_cycle += 2 # Allow two more stops in the cycle if demand is low (as we are towards the end of the scheduling process)
    elif total_demand_remaining < bus_capacity * 8:
        max_intermediate_stops_in_cycle += 3 # Allow three more stops in the cycle if demand is low (as we are towards the end of the scheduling process)
    elif total_demand_remaining < bus_capacity * 3:
        max_intermediate_stops_in_cycle += 4 # Allow four more stops in the cycle if demand is low (as we are towards the end of the scheduling process)
        # max_intermediate_stops_in_cycle = len(active_stops) # Allow all stops to be used in the cycle when demand is low (towards the end of the scheduling process)

    best_daily_cycle_info = None
    best_metric_so_far_val = (-1, -float("inf"), -float("inf"), -float("inf"))
    
    for num_stops_in_cycle_combo in range(1, min(len(active_stops), max_intermediate_stops_in_cycle) + 1):
        for stop_combination_tuple in itertools.combinations(active_stops, num_stops_in_cycle_combo):
            # Flatten the tuple of lists into a single list of stops
            stop_combination_list = [stop for sublist in stop_combination_tuple for stop in sublist]

            # Remove any potential duplicates 
            stop_combination_list = list(set(stop_combination_list))

            cycle_components = calculate_optimized_cycle_components(grocery_store, stop_combination_list, matrix_durations, location_to_idx_map)
            if not cycle_components:
                continue
            
            time_gs_to_fs = cycle_components["time_gs_to_fs"]
            
            tsp_pickup_time_from_fs = cycle_components["tsp_pickup_time_from_fs"]
            if tsp_pickup_time_from_fs > max_pickup_time_from_fs:
                continue
            
            tsp_dropoff_time_to_fs = cycle_components["tsp_dropoff_time_to_fs"]
            time_fs_to_gs = cycle_components["time_fs_to_gs"]
            core_loop_service_time = tsp_pickup_time_from_fs + STORE_WAIT_TIME_MINUTES + tsp_dropoff_time_to_fs
            core_loop_service_time += 2* (len(stop_combination_list)) * STOP_WAIT_TIME_MINUTES # Add 2 wait time for each stop in the cycle (one for pickup and one for dropoff)
            if core_loop_service_time <= 1e-5: # Avoid division by zero or non-productive loops
                continue

            max_daily_operation_minutes = (MAX_DAILY_OPERATION_HOURS * 60) + 1e-5
            max_num_round_trips_for_day = math.floor((max_daily_operation_minutes - time_gs_to_fs - time_fs_to_gs) / (core_loop_service_time))
            max_pax_cap_for_day = max_num_round_trips_for_day * bus_capacity
            total_demand_in_cycle_stops = sum(s.weekly_demand_remaining for s in stop_combination_list if s.weekly_demand_remaining > 0)
            daily_throughput = min(max_pax_cap_for_day, total_demand_in_cycle_stops)
            num_round_trips_for_day = math.ceil(daily_throughput / bus_capacity)
            total_operational_time_for_day = time_gs_to_fs + num_round_trips_for_day * core_loop_service_time + time_fs_to_gs

            current_cycle_metric_val = (daily_throughput, -total_operational_time_for_day, -len(stop_combination_list), -core_loop_service_time)

            if current_cycle_metric_val > best_metric_so_far_val:
                best_daily_cycle_info = cycle_components
                best_daily_cycle_info["daily_throughput"] = daily_throughput
                best_daily_cycle_info["num_round_trips_for_day"] = num_round_trips_for_day
                best_daily_cycle_info["total_daily_operational_time"] = total_operational_time_for_day
                best_metric_so_far_val = current_cycle_metric_val

    if best_daily_cycle_info is None:
        logger.warning("No valid cycle found for the given parameters.")
        return None

    logger.info(f"Best Daily Cycle Info: {best_daily_cycle_info}")

    # temp_cycle_demands = {s.id: s.weekly_demand_remaining for s in best_stop_comination_list}
    # best_cycle_stopid_to_stop_dict = {s.id:s for s in best_stop_comination_list}

    passengers_served_on_each_trip = []
    passengers_served_per_stop_on_each_trip = []

    sorted_stop_list = best_daily_cycle_info["tsp_pickup_path_from_fs_list"][:-1]

    logger.info(f"allocating passengers to stops:")
    logger.info(f"sorted_stop_list: {[s.id for s in sorted_stop_list]}")
    logger.info(f"weekly_demand_remaining: {[s.weekly_demand_remaining for s in sorted_stop_list]}")
    logger.info(f"bus_capacity: {bus_capacity}")
    logger.info(f"Entering loop for {best_daily_cycle_info['num_round_trips_for_day']} loops...")

    for round_trip_num in range(best_daily_cycle_info["num_round_trips_for_day"]):
        logger.info(f"\nRound Trip number: {round_trip_num + 1}")
        eligible_stops_for_this_trip = [s for s in sorted_stop_list if s.weekly_demand_remaining > 0]
        if not eligible_stops_for_this_trip:
            break 

        total_weekly_demand_remaining = sum(s.weekly_demand_remaining for s in eligible_stops_for_this_trip)
        logger.info(f"total_weekly_demand_remaining in the beginning of the run: {total_weekly_demand_remaining}")

        passengers_to_board_this_run = min(bus_capacity, total_weekly_demand_remaining)
        logger.info(f"passengers_to_board_this_run: {passengers_to_board_this_run}")

        logger.info(f"stops with demand remaining in the beginning of the run: {[s.id for s in eligible_stops_for_this_trip]}")
        logger.info(f"weekly_demand_remaining in the beginning of the run: {[s.weekly_demand_remaining for s in eligible_stops_for_this_trip]}")

        passengers_served_per_stop_this_trip = {s.id: 0 for s in sorted_stop_list}
        stop_fractions_remaining = []

        for stop_obj in eligible_stops_for_this_trip:

            desired_pickup = (stop_obj.weekly_demand_remaining / total_weekly_demand_remaining) * passengers_to_board_this_run
            pickup_amount = math.floor(desired_pickup)

            stop_fractions_remaining.append((stop_obj, desired_pickup - pickup_amount))

            passengers_served_per_stop_this_trip[stop_obj.id] = pickup_amount
            stop_obj.add_service(pickup_amount)
        
        stop_fractions_remaining.sort(key=lambda x: x[1], reverse=True)

        remaining_to_allocate = passengers_to_board_this_run - sum(passengers_served_per_stop_this_trip.values())

        for stop_obj, _ in stop_fractions_remaining:

            if remaining_to_allocate == 0:
                break
            if stop_obj.weekly_demand_remaining > 0:
                passengers_served_per_stop_this_trip[stop_obj.id] += 1
                stop_obj.add_service(1)
                remaining_to_allocate -=1
        
        if remaining_to_allocate > 0:
            logger.warning(f"Unable to allocate all passengers for this run. Remaining: {remaining_to_allocate}")

        logger.info(f"weekly_demand_remaining at the end of the run: {[s.weekly_demand_remaining for s in eligible_stops_for_this_trip]}")
        logger.info(f"passengers_served_per_stop_this_trip: {[v for k, v in passengers_served_per_stop_this_trip.items()]}")

        total_boarded_this_trip = sum(passengers_served_per_stop_this_trip.values())

        if total_boarded_this_trip != passengers_to_board_this_run:
            logger.warning(f"Total boarded this run ({total_boarded_this_trip}) does not match expected ({passengers_to_board_this_run}).")
            if total_boarded_this_trip > passengers_to_board_this_run:
                logger.warning(f"Excess passengers boarded: {total_boarded_this_trip - passengers_to_board_this_run}")
            elif total_boarded_this_trip < passengers_to_board_this_run:
                logger.warning(f"Insufficient passengers boarded: {passengers_to_board_this_run - total_boarded_this_trip}")
        else:
            logger.info(f"Total boarded this run matches expected: {total_boarded_this_trip}")

        total_weekly_demand_remaining_end = sum(s.weekly_demand_remaining for s in eligible_stops_for_this_trip)

        logger.info(f"total_weekly_demand_remaining at the end of the run: {total_weekly_demand_remaining_end}")
        diff_weekly_demand_remaining = total_weekly_demand_remaining - total_weekly_demand_remaining_end
        logger.info(f"difference in total_weekly_demand_remaining: {diff_weekly_demand_remaining}")

        if diff_weekly_demand_remaining != total_boarded_this_trip:
            logger.warning(f"Difference in total weekly demand remaining ({diff_weekly_demand_remaining}) does not match total boarded ({total_boarded_this_trip}).")
            if diff_weekly_demand_remaining > total_boarded_this_trip:
                logger.warning(f"weekly_demand_remaining excess change: {diff_weekly_demand_remaining - total_boarded_this_trip}")
            elif diff_weekly_demand_remaining < total_boarded_this_trip:
                logger.warning(f"weekly_demain_remaining insufficient change: {total_boarded_this_trip - diff_weekly_demand_remaining}")

        if total_boarded_this_trip == 0:
            break 
        
        passengers_served_on_each_trip.append(total_boarded_this_trip)
        passengers_served_per_stop_on_each_trip.append(passengers_served_per_stop_this_trip)
    
    best_daily_cycle_info['passengers_served_per_stop_on_each_trip'] = passengers_served_per_stop_on_each_trip
    best_daily_cycle_info['passengers_served_on_each_trip'] = passengers_served_on_each_trip

    if best_daily_cycle_info['daily_throughput'] != sum(passengers_served_on_each_trip):
        logger.warning(f"Daily throughput ({best_daily_cycle_info['daily_throughput']}) does not match sum of passengers served ({sum(passengers_served_on_each_trip)}).")

    first_pickup_leg_ordered_location_ids = [loc.id for loc in [grocery_store] + best_daily_cycle_info["tsp_pickup_path_from_fs_list"]]
    first_pickup_leg_time = best_daily_cycle_info["time_gs_to_fs"] + best_daily_cycle_info["tsp_pickup_time_from_fs"]

    middle_pickup_leg_ordered_location_ids = [loc.id for loc in best_daily_cycle_info["tsp_pickup_path_from_fs_list"]]
    middle_pickup_leg_time = best_daily_cycle_info["tsp_pickup_time_from_fs"]

    middle_dropoff_leg_ordered_location_ids = [loc.id for loc in best_daily_cycle_info["tsp_dropoff_path_to_fs_list"]]
    middle_dropoff_leg_time = best_daily_cycle_info["tsp_dropoff_time_to_fs"]

    last_dropoff_leg_ordered_location_ids = [loc.id for loc in best_daily_cycle_info["tsp_dropoff_path_to_fs_list"] + [grocery_store]]
    last_dropoff_leg_time = best_daily_cycle_info["tsp_dropoff_time_to_fs"] + best_daily_cycle_info["time_fs_to_gs"]

    num_round_trips_for_day = best_daily_cycle_info.get("num_round_trips_for_day", 0)
    passengers_served_on_each_trip = best_daily_cycle_info.get("passengers_served_on_each_trip", [])
    passengers_served_per_stop_on_each_trip = best_daily_cycle_info.get("passengers_served_per_stop_on_each_trip", [])

    if num_round_trips_for_day == 0:
        logger.warning("No loops for the day. Returning None.")
        return None

    first_pickup_leg = RouteLeg(
        ordered_location_ids=first_pickup_leg_ordered_location_ids,
        travel_time_minutes=first_pickup_leg_time,
        passengers_serviced=passengers_served_on_each_trip[0],
        passengers_served_per_stop=passengers_served_per_stop_on_each_trip[0]
    )

    last_dropoff_leg = RouteLeg(
        ordered_location_ids=last_dropoff_leg_ordered_location_ids,
        travel_time_minutes=last_dropoff_leg_time,
        passengers_serviced=passengers_served_on_each_trip[-1],
        passengers_served_per_stop=passengers_served_per_stop_on_each_trip[-1]
    )

    if num_round_trips_for_day == 1:
        round_trip = RoundTrip(
            pickup_leg=first_pickup_leg,
            dropoff_leg=last_dropoff_leg
        )
        return [round_trip]
    else:
        round_trips = []
        for round_trip_num in range(num_round_trips_for_day):
            middle_pickup_leg = RouteLeg(
                ordered_location_ids=middle_pickup_leg_ordered_location_ids,
                travel_time_minutes=middle_pickup_leg_time,
                passengers_serviced=passengers_served_on_each_trip[round_trip_num],
                passengers_served_per_stop=passengers_served_per_stop_on_each_trip[round_trip_num]
            )
            middle_dropoff_leg = RouteLeg(
                ordered_location_ids=middle_dropoff_leg_ordered_location_ids,
                travel_time_minutes=middle_dropoff_leg_time,
                passengers_serviced=passengers_served_on_each_trip[round_trip_num],
                passengers_served_per_stop=passengers_served_per_stop_on_each_trip[round_trip_num]
            )

            if round_trip_num == 0:
                round_trip = RoundTrip(
                    pickup_leg=first_pickup_leg,
                    dropoff_leg=middle_dropoff_leg
                )
            elif round_trip_num == num_round_trips_for_day - 1:
                round_trip = RoundTrip(
                    pickup_leg=middle_pickup_leg,
                    dropoff_leg=last_dropoff_leg
                )
            else:
                round_trip = RoundTrip(
                    pickup_leg=middle_pickup_leg,
                    dropoff_leg=middle_dropoff_leg
                )
            round_trips.append(round_trip)

    return round_trips


# def total_demand_remaining(list_of_stops):
#     return sum(stop.weekly_demand_remaining for stop in list_of_stops)


def run_simulation(
        initial_grocery_stores, 
        initial_transit_stops, 
        mapbox_durations_matrix, 
        location_to_idx_map, 
        bus_capacity=15, 
        logger=LOGGER, 
        max_intermediate_stops_in_cycle=3,
        max_pickup_time_from_fs=60,
        outer_recalibration_loops=50,
        inner_recalibration_loops=10,
        transit_stop_clusters={}):
    # ... (rest of run_simulation needs to be updated to use the new best_daily_cycle_info structure)
    # This will be the next step.
    
    if not initial_transit_stops:
        return 0, 0, {}, []
    max_buses_to_try = len(initial_transit_stops) * 3

    # Cluster transit stops before simulation
    ts_id_list = [ts.id for ts in initial_transit_stops]
    
    # Make a deep copy of transit stops for this simulation attempt to track weekly demand
    global temp_ts_id_to_ts_dict, main_ts_id_to_ts_dict
    main_ts_id_to_ts_dict = {ts.id: ts for ts in initial_transit_stops}
    ts_id_list = [[ts.id] for ts in initial_transit_stops]

    # temp_transit_stops = copy.deepcopy(initial_transit_stops)
    # temp_ts_id_to_ts_dict = {ts.id: ts for ts in temp_transit_stops}
    # # current_sim_transit_stops_global = copy.deepcopy(initial_transit_stops)

    total_demand = sum(s.total_demand for s in main_ts_id_to_ts_dict.values())
    remaining_demand_overall = sum(s.weekly_demand_remaining for s in main_ts_id_to_ts_dict.values())

    if total_demand != remaining_demand_overall:
        logger.warning(f"Total demand ({total_demand}) does not match remaining demand overall in the beginning of simulation ({remaining_demand_overall}).")
        return -1, float("inf"), {}, []
    else:
        logger.info(f"Total demand matches remaining demand overall in the beginning of simulation: {total_demand} == {remaining_demand_overall}")

    logger.info(f"remaining_demand_overall in the beginning before the loop: {remaining_demand_overall}")

    buses = []
    bus_idx = 1
    best_ts_id_to_ts_dict = main_ts_id_to_ts_dict

    while remaining_demand_overall > 0: 

        if bus_idx > max_buses_to_try:
            logger.warning("Maximum number of busses reached. No more buses to try.")
            return -1, float("inf"), {}, []       

        logger.info(f"Working on bus # {bus_idx} ...")

        best_remaining_demand_overal = sum(s.weekly_demand_remaining for s in best_ts_id_to_ts_dict.values())
        best_bus_total_weekly_driving_time = float("inf")
        best_bus_total_passengers_served = -float("inf")
        best_metric = (best_remaining_demand_overal, best_bus_total_weekly_driving_time, -best_bus_total_passengers_served)

        for grocery_store in initial_grocery_stores:
            temp_ts_id_to_ts_dict = copy.deepcopy(main_ts_id_to_ts_dict)
            temp_bus = Bus(f"Bus-{bus_idx}", capacity=bus_capacity)
            temp_bus.reset_for_new_week()
            temp_bus.assigned_grocery_store_id = grocery_store.id
            temp_bus.assigned_grocery_store = grocery_store

            for day in range(5):
                daily_round_trip_list = find_best_daily_cycle(
                    grocery_store, 
                    ts_id_list,
                    bus_capacity,
                    mapbox_durations_matrix, 
                    location_to_idx_map, 
                    max_intermediate_stops_in_cycle=max_intermediate_stops_in_cycle,
                    logger=logger,
                    reference_ts_dict="temp",
                    max_pickup_time_from_fs=max_pickup_time_from_fs
                )

                if daily_round_trip_list is None:
                    logger.info(f"No valid daily cycle found for Bus {temp_bus.id} on day {day}.")
                    continue

                for round_trip in daily_round_trip_list:
                    if temp_bus.add_trip(day, round_trip):
                        logger.info(f"Added trip for Bus {temp_bus.id} on day {day}.")
                    else:
                        logger.warning(f"Failed to add trip for Bus {temp_bus.id} on day {day}.")
                
            flat_ts_id_list = [ts_id for sublist in ts_id_list for ts_id in sublist]
            temp_transit_stops = [temp_ts_id_to_ts_dict[ts_id] for ts_id in flat_ts_id_list]
            temp_remaining_demand_overall = sum(s.weekly_demand_remaining for s in temp_transit_stops)
            temp_bus_total_weekly_driving_time = temp_bus.total_weekly_driving_time
            temp_bus_total_passengers_served = temp_bus.total_weekly_passengers_served

            temp_metric = (temp_remaining_demand_overall, temp_bus_total_weekly_driving_time, -temp_bus_total_passengers_served)

            if temp_metric < best_metric:
                best_bus = temp_bus
                best_ts_id_to_ts_dict = temp_ts_id_to_ts_dict
                best_metric = temp_metric
        
        if remaining_demand_overall == best_metric[0]:
            logger.warning("No bus can cover the remaining demand given the current constraints.")
            return -1, float("inf"), {}, []
        
        demand_covered_for_these_stops = remaining_demand_overall - best_metric[0]
        total_boarded_to_this_bus = sum(
            n_pax
            for schedule_value in best_bus.daily_schedule.values()
            for round_trip in schedule_value
            for n_pax in round_trip.pickup_leg.passengers_served_per_stop.values()
        )

        if total_boarded_to_this_bus != demand_covered_for_these_stops:
            logger.warning(f"Total boarded to this bus ({total_boarded_to_this_bus}) does not match demand covered to these stops ({demand_covered_for_these_stops}).")
            if total_boarded_to_this_bus > demand_covered_for_these_stops:
                logger.warning(f"Excess passengers boarded: {total_boarded_to_this_bus - demand_covered_for_these_stops}")
            elif total_boarded_to_this_bus < demand_covered_for_these_stops:
                logger.warning(f"Insufficient passengers boarded: {demand_covered_for_these_stops - total_boarded_to_this_bus}")
        else:
            logger.info(f"Total boarded to this bus matches demand covered to these stops: {total_boarded_to_this_bus}")

        if remaining_demand_overall == best_metric[0]:
            logger.warning("No bus can cover the remaining demand given the current constraints.")
            return -1, float("inf"), {}, []
        
        remaining_demand_overall = best_metric[0]
        buses.append(best_bus)
        main_ts_id_to_ts_dict = best_ts_id_to_ts_dict
        bus_idx += 1

        total_boarded_to_all_buses = sum(
            n_pax
            for bus in buses
            for schedule_value in bus.daily_schedule.values()
            for round_trip in schedule_value
            for n_pax in round_trip.pickup_leg.passengers_served_per_stop.values()
        )
        total_to_be_boarded = total_demand - total_boarded_to_all_buses
        if remaining_demand_overall != total_to_be_boarded:
            logger.warning(f"After scheduling {best_bus.id}, total demand remaining ({remaining_demand_overall}) does not match total to be boarded ({total_to_be_boarded}).")
            if remaining_demand_overall > total_to_be_boarded:
                logger.warning(f"Excess demand remaining: {remaining_demand_overall - total_to_be_boarded}")
            elif remaining_demand_overall < total_to_be_boarded:
                logger.warning(f"Insufficient demand remaining: {total_to_be_boarded - remaining_demand_overall}")
        else:
            logger.info(f"After scheduling {best_bus.id}, total demand remaining matches total to be boarded: {remaining_demand_overall}")
        
    
    best_solution_number_of_buses_to_deploy = len(buses)
    best_solution_total_weekly_driving_time = sum(bus.total_weekly_driving_time for bus in buses)
    
    best_solution_schedule_output_data = {}
    for bus_obj in buses:
        best_solution_schedule_output_data[bus_obj.id] = bus_obj.daily_schedule

    best_buses = copy.deepcopy(buses)
    best_bus_total_weekly_driving_time = sum(bus.total_weekly_driving_time for bus in best_buses)
    for i in range(outer_recalibration_loops):
        temp_buses = copy.deepcopy(buses)
        for j in range(inner_recalibration_loops):
            (
                temp_buses,
                recalibration_status
            ) = recalibrate( 
                temp_buses,
                initial_grocery_stores,
                mapbox_durations_matrix,
                location_to_idx_map,
                logger=logger,
                max_intermediate_stops_in_cycle=max_intermediate_stops_in_cycle,
                max_pickup_time_from_fs=max_pickup_time_from_fs,
                transit_stop_clusters=transit_stop_clusters
            )
            if recalibration_status == 1:
                temp_bus_total_weekly_driving_time = sum(bus.total_weekly_driving_time for bus in temp_buses)
                if temp_bus_total_weekly_driving_time < best_bus_total_weekly_driving_time:
                    best_buses = copy.deepcopy(temp_buses)
                    best_bus_total_weekly_driving_time = temp_bus_total_weekly_driving_time
                    logger.info(f"Found a better solution with total weekly driving time: {best_bus_total_weekly_driving_time}")
    
    best_solution_number_of_buses_to_deploy = len(best_buses)
    best_solution_total_weekly_driving_time = sum(bus.total_weekly_driving_time for bus in best_buses)
    return (
        best_solution_number_of_buses_to_deploy, 
        best_solution_total_weekly_driving_time, 
        best_solution_schedule_output_data, 
        best_buses,
    )


def reset_days_to_recalibrate(bus_objects_list, reference_ts_dict="main"):
    bus_days_to_recalibrate = {}
    stop_ids_to_recalibrate = set()
    for bus in bus_objects_list:
        bus.mark_days_to_recalibrate()
        days_to_recalibrate = sorted(set([(bus.id, day) for day in bus.days_to_recalibrate]), key=lambda x: x[1])
        bus_days_to_recalibrate[bus.id] = days_to_recalibrate
        for _, day in days_to_recalibrate:
            stop_ids_to_recalibrate |= (bus.reset_day_for_recalibration(day, reference_ts_dict=reference_ts_dict))
        if len(days_to_recalibrate) == 5:
            bus.reset_for_new_week()
    
    return bus_days_to_recalibrate, list(stop_ids_to_recalibrate)


def get_transit_stop_ids_from_buses(buses):
    """Extract community centers from the trucks' schedules"""
    transit_stop_ids = set()
    # Iterate through all trucks and their daily schedules
    for bus in buses:
        for day, trips in bus.daily_schedule.items():
            for trip in trips:
                for loc_ids in trip.pickup_leg.ordered_location_ids:
                    if loc_ids in trip.pickup_leg.passengers_served_per_stop:
                        transit_stop_ids.add(loc_ids)
    return transit_stop_ids


def randomly_cluster_stop_ids_for_recalibration(
        stop_ids_to_recalibrate,
        transit_stop_clusters,
        logger=LOGGER):
    
    logger.debug(f"transit_stop_clusters: {json.dumps(transit_stop_clusters, indent=2)}")
    logger.debug(f"stop_ids_to_recalibrate: {json.dumps(stop_ids_to_recalibrate, indent=2)}")
    used_stop_ids = []
    stop_id_clusters = []
    # Randomly shuffle the stop IDs for recalibration
    random.shuffle(stop_ids_to_recalibrate)
    logger.debug(f"shuffled stop_ids_to_recalibrate: {json.dumps(stop_ids_to_recalibrate, indent=2)}")
    for stop_id in stop_ids_to_recalibrate:
        logger.debug(f"stop_id: {stop_id}")
        logger.debug(f"used_stop_ids: {used_stop_ids}")

        if stop_id in used_stop_ids:
            logger.debug(f"stop_id {stop_id} already used in a cluster. Skipping.")
            continue
        flag = True
        while flag:
            n_cluster = random.choice(list(transit_stop_clusters.keys())) if transit_stop_clusters else None
            logger.debug(f"n_cluster: {n_cluster}")
            if n_cluster is not None:
                clustering = transit_stop_clusters[n_cluster]
                logger.debug(f"clustering: {json.dumps(clustering, indent=2)}")
                for label, cluster in clustering.items():
                    if stop_id in cluster:
                        break
                logger.debug(f"Found cluster for stop_id {stop_id}: {json.dumps(cluster, indent=2)}")
                if not any(stop_id in used_stop_ids for stop_id in cluster):
                    for stop_id in cluster:
                        used_stop_ids.append(stop_id)
                    logger.debug(f"used_stop_ids after adding this cluster's items: {json.dumps(used_stop_ids, indent=2)}")
                    stop_id_clusters.append(cluster)
                    logger.debug(f"stop_id_clusters after adding this cluster: {json.dumps(stop_id_clusters, indent=2)}")
                    flag = False
                else:
                    logger.debug(f"Some stop_ids in cluster {json.dumps(cluster, indent=2)} are already used. Skipping this cluster.")
    
    return stop_id_clusters


def recalibrate( 
        current_buses,
        initial_grocery_stores, 
        mapbox_durations_matrix, 
        location_to_idx_map, 
        logger=LOGGER,
        max_intermediate_stops_in_cycle=3,
        max_pickup_time_from_fs=60,
        transit_stop_clusters={}):
    
    if any(bus.assigned_grocery_store is None for bus in current_buses):
        logger.warning("Some buses do not have an assigned grocery store. Cannot recalibrate.")
        return current_buses, 2
    else:
        logger.info("All buses have an assigned grocery store. Proceeding with recalibration.")
    
    current_total_weekly_driving_time = sum(bus.total_weekly_driving_time for bus in current_buses)
    
    # Make copies to avoid modifying originals
    new_buses = copy.deepcopy(current_buses)
    bus_id_to_bus = {bus.id: bus for bus in new_buses}
    new_stop_ids = get_transit_stop_ids_from_buses(new_buses)

    global temp_ts_id_to_ts_dict, main_ts_id_to_ts_dict
    temp_ts_id_to_ts_dict = copy.deepcopy(main_ts_id_to_ts_dict)

    new_stops = [temp_ts_id_to_ts_dict[stop_id] for stop_id in new_stop_ids]

    passengers_served_by_buses = sum(trip.pickup_leg.passengers_serviced for bus in new_buses for day, trips in bus.daily_schedule.items() for trip in trips)
    total_demand_of_all_stops = sum(s.total_demand for s in new_stops)
    remaining_demand_all = sum(s.weekly_demand_remaining for s in new_stops)
    passengers_served_at_all_stops = total_demand_of_all_stops - remaining_demand_all
    if passengers_served_by_buses != passengers_served_at_all_stops:
        logger.warning(f"Before reseting days to recalibrate, passengers served by buses ({passengers_served_by_buses}) does not match passengers served at all stops ({passengers_served_at_all_stops}).")
    

    bus_days_to_recalibrate, stop_ids_to_recalibrate = reset_days_to_recalibrate(new_buses, reference_ts_dict="temp")


    if len(stop_ids_to_recalibrate) == 0:
        logger.info("No need to recalibrate!")
        recalibration_status = 3 # No need to recalibrate
        return current_buses, recalibration_status
    
    stops_to_recalibrate = [temp_ts_id_to_ts_dict[stop_id] for stop_id in stop_ids_to_recalibrate]
    remaining_demand_overall = sum(s.weekly_demand_remaining for s in stops_to_recalibrate)
    logger.info(f"remaining_demand_overall in the beginning before recalibration: {remaining_demand_overall}")
    remaining_demand_all = sum(s.weekly_demand_remaining for s in new_stops)
    if remaining_demand_overall != remaining_demand_all:
        logger.warning(f"After reseting days to recalibrate, remaining demand at stops to recalibrate ({remaining_demand_overall}) does not match remaining demand at all stops ({remaining_demand_all}).")
    
    total_demand_of_all_stops = sum(s.total_demand for s in new_stops)
    total_demand_of_stops_to_recalibrate = sum(s.total_demand for s in stops_to_recalibrate)
    logger.info(f"total_demand_of_stops_to_recalibrate: {total_demand_of_stops_to_recalibrate}")

    demand_covered_at_all_stops = total_demand_of_all_stops - remaining_demand_all
    passengers_served_by_buses = sum(trip.pickup_leg.passengers_serviced for bus in new_buses for day, trips in bus.daily_schedule.items() for trip in trips)

    if demand_covered_at_all_stops != passengers_served_by_buses:
        logger.warning(f"After reseting days to recalibrate, demand covered at all stops ({demand_covered_at_all_stops}) does not match passengers served by buses ({passengers_served_by_buses}).")
        # logger.warning(f"total_demand_of_all_stops: {total_demand_of_all_stops}")
        # logger.warning(f"remaining_demand_all: {remaining_demand_all}")
    


    clustered_stop_ids_to_recalibrate = randomly_cluster_stop_ids_for_recalibration(
        stop_ids_to_recalibrate,
        transit_stop_clusters=transit_stop_clusters,
        logger=logger
    )
    # clustered_stop_ids_to_recalibrate = [v for k, v in transit_stop_clusters[0].items()]
    logger.warning(f"clustered_stop_ids_to_recalibrate: {json.dumps(clustered_stop_ids_to_recalibrate, indent=2)}")

    random_schedule_order = random_weave(bus_days_to_recalibrate)

    new_total_weekly_driving_time = 0
    for bus_id, day in random_schedule_order:

        logger.info(f"Recalibrating bus {bus_id} on day {day + 1}")

        bus = bus_id_to_bus[bus_id]
        
        bus_grocery_store = bus.get_assigned_grocery_store()

        if bus_grocery_store is None:
            # asssign a random grocery store from the list of initial grocery stores
            logger.info(f"Bus {bus.id} assigned grocery store is being reset")
            bus_grocery_store = random.choice(initial_grocery_stores)
            bus.assigned_grocery_store_id = bus_grocery_store.id
            bus.assigned_grocery_store = bus_grocery_store
        logger.info(f"Bus {bus.id} is assigned grocery store: {bus_grocery_store.name}")
            
        best_bus_daily_trips = find_best_daily_cycle(
            grocery_store=bus_grocery_store, 
            ts_id_list=clustered_stop_ids_to_recalibrate,
            bus_capacity=bus.capacity,
            matrix_durations=mapbox_durations_matrix,
            location_to_idx_map=location_to_idx_map,
            max_intermediate_stops_in_cycle=max_intermediate_stops_in_cycle,
            logger=logger,
            reference_ts_dict="temp",
            max_pickup_time_from_fs=max_pickup_time_from_fs
        )

        if not best_bus_daily_trips:
            logger.info(f"Did not find a best cycle for Day {day + 1}")
            continue
        
        logger.info(f"Found a best cycle for Day {day + 1}")

        for round_trip in best_bus_daily_trips:
            if bus.add_trip(day, round_trip):
                logger.info(f"Added trip for Bus {bus.id} on day {day + 1}.")
            else:
                logger.warning(f"Failed to add trip for Bus {bus.id} on day {day + 1}. Trip exceeds time limit.")
    
    stops_to_recalibrate = [temp_ts_id_to_ts_dict[stop_id] for stop_id in stop_ids_to_recalibrate]
    remaining_demand_overall = sum(s.weekly_demand_remaining for s in stops_to_recalibrate)
    new_total_weekly_driving_time = sum(bus.total_weekly_driving_time for bus in new_buses)
    
    logger.info(f"New total weekly driving time after recalibration: {new_total_weekly_driving_time}")
    logger.info(f"Current total weekly driving time: {current_total_weekly_driving_time}")
    logger.info(f"Remaining demand overall after recalibration: {remaining_demand_overall}")

    if new_total_weekly_driving_time >= current_total_weekly_driving_time or remaining_demand_overall > 0:
        logger.info(f"Recalibration did not improve the solution.")
        recalibration_status = 2 # Recalibration did not improve the solution
        return current_buses, recalibration_status
    
    logger.info(f"Calibration improved the solution.")
    recalibration_status = 1 # Calibration improved the solution
    # Check if there is any obvious overutilization of the buses:
    gs2bus_list = {}
    update_bus_list = False
    for bus in new_buses:
        gs = bus.get_assigned_grocery_store()
        bus_capacity = bus.capacity
        n_empty_days = len([day for day in bus.daily_schedule if len(bus.daily_schedule[day]) == 0])
        if (gs.id, bus_capacity) not in gs2bus_list:
            gs2bus_list[(gs.id, bus_capacity)] = []
        gs2bus_list[(gs.id, bus_capacity)].append((bus, n_empty_days))
    for (gs_id, bus_capacity), bus_list in gs2bus_list.items():
        total_empty_days = sum([n_empty_days for bus, n_empty_days in bus_list]) 
        while total_empty_days > 4:
            bus_list.sort(key=lambda x: x[1], reverse=False)
            bus_to_drop, n_empty_days = bus_list.pop()
            logger.info(f"Bus {bus_to_drop.id} is being dropped from grocery store {gs_id} as its service can be covered by other buses of same capacity assigned to this grocery store.")
            update_bus_list = True
            other_empty_bus_days = []
            for bus, n_empty_days in bus_list:
                for day in range(5-n_empty_days, 5):
                    other_empty_bus_days.append((bus, day))

            daily_schedule = bus_to_drop.daily_schedule
            for day in range(4,0,-1):
                if len(daily_schedule[day]) > 0:
                    other_bus, day_to_reassign = other_empty_bus_days.pop(0)
                    for round_trip in daily_schedule[day]:
                        other_bus.add_trip(day_to_reassign, round_trip)

            total_empty_days = sum([n_empty_days for bus, n_empty_days in bus_list])

    if update_bus_list:
        new_buses = [bus for gs_id, bus_list in gs2bus_list.items() for bus, n_empty_days in bus_list]
        new_buses.sort(key=lambda x: x.id)

    
    return new_buses, recalibration_status