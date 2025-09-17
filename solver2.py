import pandas as pd
from ortools.sat.python import cp_model
import datetime
import os
import time

# --- Configurable Metro Lines ---
METRO_LINES = {
    "Line A (Short: 20km)": 250,    # Approx. 20km route, multiple trips
    "Line B (Medium: 40km)": 450,   # Approx. 40km route, multiple trips
    "Line C (Long: 60km)": 650,     # Approx. 60km route, multiple trips
    "Line D (Express: 80km)": 900,  # Approx. 80km route, multiple trips
    "Line E (Long Express: 100km)": 1100,  # Approx. 100km route
}

def load_data(scenario_path):
    """
    Loads all CSV files for a given scenario into a dictionary of Pandas DataFrames.
    """
    if not os.path.isdir(scenario_path):
        print(f"Error: Directory not found at '{scenario_path}'")
        return None
    data_files = { "trainsets": "trainsets_master.csv", "certificates": "fitness_certificates.csv", "job_cards": "job_cards_maximo.csv", "slas": "branding_slas.csv", "resources": "depot_resources.csv", "layout_costs": "depot_layout_costs.csv" }
    data = {}
    print("--- Loading Data ---")
    for key, filename in data_files.items():
        try:
            filepath = os.path.join(scenario_path, filename)
            data[key] = pd.read_csv(filepath)
            print(f"Loaded {len(data[key])} records from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    data["certificates"]["expiry_date"] = pd.to_datetime(data["certificates"]["expiry_date"]).dt.date
    return data

def preprocess_data_with_reasons(data):
    """
    Processes raw data to determine eligibility and provides specific reasons for ineligibility.
    """
    train_ids = data["trainsets"]["trainset_id"].tolist()
    eligibility_details = {}
    today = datetime.date(2025, 9, 18)
    print("\n--- Preprocessing Data: Checking Train Eligibility ---")
    
    for train_id in train_ids:
        eligibility_details[train_id] = {'is_eligible': True, 'reason': 'Eligible for service'}
        
        certs = data["certificates"][data["certificates"]["trainset_id"] == train_id]
        if len(certs) < 3:
            eligibility_details[train_id] = {'is_eligible': False, 'reason': 'Missing required certificates'}
            continue
        if (certs["expiry_date"] < today).any():
            eligibility_details[train_id] = {'is_eligible': False, 'reason': 'Certificate expired'}
            continue
            
        jobs = data["job_cards"][data["job_cards"]["trainset_id"] == train_id]
        critical_open_jobs = jobs[(jobs["status"] == "OPEN") & (jobs["is_critical"] == True)]
        if not critical_open_jobs.empty:
            eligibility_details[train_id] = {'is_eligible': False, 'reason': 'Critical maintenance open'}
            
    print("Eligibility checks complete.")
    return eligibility_details

def preprocess_shunting_costs(layout_df):
    """Calculates average costs for key shunting moves."""
    to_maintenance_moves = layout_df[layout_df['to_location'].str.contains('IBL_Bay', na=False)]
    avg_cost_to_maintenance = to_maintenance_moves['shunting_cost'].mean() if not to_maintenance_moves.empty else 0
    to_stabling_moves = layout_df[layout_df['to_location'].str.contains('Stabling_Track', na=False)]
    avg_cost_to_stabling = to_stabling_moves['shunting_cost'].mean() if not to_stabling_moves.empty else 0
    return {"maintenance": int(avg_cost_to_maintenance), "stabling": int(avg_cost_to_stabling)}

def solve_primary_assignment(data, eligibility_details, shunting_costs):
    """
    PHASE 1: Solves the core assignment problem with a penalty for not fixing critical trains.
    """
    start_time = time.time()
    model = cp_model.CpModel()
    train_ids = data["trainsets"]["trainset_id"].tolist()
    avg_mileage = data["trainsets"]["cumulative_mileage_km"].mean()
    print(f"\n--- Phase 1: Primary Assignment Solver ---")
    print(f"Setting up and solving model for {len(train_ids)} trains...")
    assignments = {}
    for train_id in train_ids:
        assignments[train_id] = {"service": model.NewBoolVar(f"{train_id}_service"), "standby": model.NewBoolVar(f"{train_id}_standby"), "maintenance": model.NewBoolVar(f"{train_id}_maintenance")}
        model.AddExactlyOne(list(assignments[train_id].values()))
        if not eligibility_details.get(train_id, {}).get('is_eligible', False):
            model.Add(assignments[train_id]["service"] == 0)
    ibl_bays_capacity = data["resources"].loc[data["resources"]["resource_id"] == "IBL_Bays", "available_capacity"].iloc[0]
    model.Add(sum(assignments[t]["maintenance"] for t in train_ids) <= ibl_bays_capacity)
    manpower_capacity = data["resources"].loc[data["resources"]["resource_id"] == "Cleaning_Staff_ManHours", "available_capacity"].iloc[0]
    required_hours = data["job_cards"][data["job_cards"]["status"] == "OPEN"].groupby("trainset_id")["required_man_hours"].sum().to_dict()
    model.Add(sum(assignments[t]["maintenance"] * int(required_hours.get(t, 0)) for t in train_ids) <= manpower_capacity)
    
    critically_failed_trains = [
        train_id for train_id, details in eligibility_details.items() 
        if not details['is_eligible'] and "Critical" in details['reason']
    ]
    urgency_penalty = sum(
        assignments[train_id]["standby"] for train_id in critically_failed_trains
    )

    mileage_deviations = []
    for train_id in train_ids:
        train_mileage = data["trainsets"].loc[data["trainsets"]["trainset_id"] == train_id, "cumulative_mileage_km"].iloc[0]
        dev = int(train_mileage - avg_mileage)
        abs_dev = model.NewIntVar(0, 200000, f'abs_dev_{train_id}')
        model.AddAbsEquality(abs_dev, dev)
        service_mileage_cost = model.NewIntVar(0, 200000, f'serv_mileage_cost_{train_id}')
        model.Add(service_mileage_cost == abs_dev).OnlyEnforceIf(assignments[train_id]["service"])
        model.Add(service_mileage_cost == 0).OnlyEnforceIf(assignments[train_id]["service"].Not())
        mileage_deviations.append(service_mileage_cost)
        
    total_mileage_deviation_cost = sum(mileage_deviations)
    total_branding_penalty = sum((1 - assignments[sla["trainset_id"]]["service"]) * int(sla["penalty_per_hour"]) for _, sla in data["slas"].iterrows() if sla["trainset_id"] in assignments)
    total_shunting_cost = sum(assignments[t]["maintenance"] * shunting_costs["maintenance"] + (assignments[t]["service"] + assignments[t]["standby"]) * shunting_costs["stabling"] for t in train_ids)
    
    w_mileage, w_branding, w_shunting, w_urgency = 1, 10000, 10, 1000000
    
    model.Minimize(
        w_mileage * total_mileage_deviation_cost + 
        w_branding * total_branding_penalty + 
        w_shunting * total_shunting_cost +
        w_urgency * urgency_penalty
    )
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 8
    status = solver.solve(model)
    print(f"Solving completed in {time.time() - start_time:.2f} seconds.")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = {t: ("Revenue Service" if solver.Value(assignments[t]["service"]) else "Standby" if solver.Value(assignments[t]["standby"]) else "Maintenance") for t in train_ids}
        return solution, required_hours
    else:
        print(f"Solver failed with status: {solver.StatusName(status)}")
        return None, None

def show_train_recommendations_for_line(primary_solution, eligibility_details, required_hours, data, lines):
    """
    Asks the user to select a line and shows a ranked list of all trains with a detailed dashboard view.
    """
    if not primary_solution: return

    # 1. Ask user to select a line
    print("\n" + "="*50)
    print("PHASE 2: TRAIN RECOMMENDATION TOOL")
    print("="*50)
    line_list = list(lines.keys())
    for i, line_name in enumerate(line_list, 1):
        print(f"[{i}] {line_name}")
    try:
        choice = int(input(f"\nEnter the line number to see recommendations for (1-{len(line_list)}): ").strip())
        if not 1 <= choice <= len(line_list):
            print("Invalid selection. Exiting."); return
        selected_line = line_list[choice - 1]
    except ValueError:
        print("Invalid input. Please enter a number. Exiting."); return

    # 2. Prepare detailed dashboard data for all trains
    all_trains_details = []
    avg_fleet_mileage = data["trainsets"]["cumulative_mileage_km"].mean()
    today = datetime.date(2025, 9, 18)
    
    for train_id, status in primary_solution.items():
        mileage = data["trainsets"].loc[data["trainsets"]["trainset_id"] == train_id, "cumulative_mileage_km"].iloc[0]
        mileage_vs_avg = ((mileage - avg_fleet_mileage) / avg_fleet_mileage) * 100
        
        pending_work_hrs = required_hours.get(train_id, 0)
        
        future_certs = data['certificates'][(data['certificates']['trainset_id'] == train_id) & (data['certificates']['expiry_date'] >= today)]
        next_cert_expiry = future_certs['expiry_date'].min() if not future_certs.empty else "N/A"

        all_trains_details.append({
            'id': train_id, 
            'status': status, 
            'mileage': mileage, 
            'mileage_vs_avg': mileage_vs_avg,
            'pending_work_hrs': pending_work_hrs,
            'next_cert_expiry': next_cert_expiry,
            'eligibility': eligibility_details[train_id]
        })

    # 3. Separate trains and apply sorting logic
    service_trains = [t for t in all_trains_details if t['status'] == 'Revenue Service']
    standby_trains = [t for t in all_trains_details if t['status'] == 'Standby']
    maintenance_trains = [t for t in all_trains_details if t['status'] == 'Maintenance']
    
    avg_line_mileage = sum(lines.values()) / len(lines)
    is_long_line = lines[selected_line] >= avg_line_mileage
    
    # For long lines, we need trains with low mileage. Branding needs are a secondary tie-breaker.
    # The branding needs calculation is now part of the reasoning, not the sort.
    if is_long_line:
        print("\n(Sorting for a LONG line: Low Mileage is better)")
        service_trains.sort(key=lambda x: x['mileage'])
    else:
        print("\n(Sorting for a SHORT line: High Mileage is better)")
        service_trains.sort(key=lambda x: x['mileage'], reverse=True)

    standby_trains.sort(key=lambda x: x['mileage'])
    maintenance_trains.sort(key=lambda x: x['mileage'])
    final_ranked_list = service_trains + standby_trains + maintenance_trains
    
    # 4. Display the final dashboard
    print("\n" + "="*125)
    print(f"--- Full Fleet Dashboard ranked for: {selected_line} ---")
    print("="*125)
    print(f"{'Rank':<6} {'Train ID':<10} {'Status':<18} {'Mileage vs. Avg (%)':<22} {'Next Cert Expiry':<20} {'Pending Work (hrs)':<22} {'Reasoning'}")
    print("-" * 125)
    
    for i, train in enumerate(final_ranked_list, 1):
        reason = ""
        if not train['eligibility']['is_eligible']:
            reason = train['eligibility']['reason']
        elif train['status'] == 'Revenue Service':
            mileage_status = "Low Mileage" if train['mileage'] < avg_fleet_mileage else "High Mileage"
            reason = f"Good for this route ({mileage_status})"
        elif train['status'] == 'Maintenance':
            reason = f"Scheduled work ({int(train['pending_work_hrs'])} hrs)"
        elif train['status'] == 'Standby':
            if train['mileage'] > avg_fleet_mileage:
                reason = "Eligible, but high mileage (held for balancing)"
            else:
                reason = "Eligible, held as operational spare"

        expiry_str = str(train['next_cert_expiry']) if train['next_cert_expiry'] != "N/A" else "N/A"
        print(f"#{i:<5} {train['id']:<10} {train['status']:<18} {train['mileage_vs_avg']:<+22.1f} {expiry_str:<20} {int(train['pending_work_hrs']):<22} {reason}")
    print("-" * 125)

if __name__ == "__main__":
    SCENARIO_FOLDER = 'bottleneck_case' 
    
    data_frames = load_data(SCENARIO_FOLDER)
    if data_frames:
        eligibility_details = preprocess_data_with_reasons(data_frames)
        shunting_costs_dict = preprocess_shunting_costs(data_frames["layout_costs"])
        
        primary_solution, required_hours = solve_primary_assignment(data_frames, eligibility_details, shunting_costs_dict)
        
        if primary_solution:
            print(f"\nPhase 1 Complete. Identified {len([s for s in primary_solution.values() if s == 'Revenue Service'])} trains fit for service.")
            
            show_train_recommendations_for_line(primary_solution, eligibility_details, required_hours, data_frames, METRO_LINES)
        else:
            print("No solution found for the primary assignment.")