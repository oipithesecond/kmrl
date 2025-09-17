# solver.py
import pandas as pd
from ortools.sat.python import cp_model
import datetime
import os

def load_data(scenario_path):
    """
    Loads all CSV files for a given scenario into a dictionary of Pandas DataFrames.
    """
    if not os.path.isdir(scenario_path):
        print(f"Error: Directory not found at '{scenario_path}'")
        return None

    data = {
        "trainsets": pd.read_csv(os.path.join(scenario_path, "trainsets_master.csv")),
        "certificates": pd.read_csv(os.path.join(scenario_path, "fitness_certificates.csv")),
        "job_cards": pd.read_csv(os.path.join(scenario_path, "job_cards_maximo.csv")),
        "slas": pd.read_csv(os.path.join(scenario_path, "branding_slas.csv")),
        "resources": pd.read_csv(os.path.join(scenario_path, "depot_resources.csv")),
        "layout_costs": pd.read_csv(os.path.join(scenario_path, "depot_layout_costs.csv")),
    }
    
    # Convert date column to datetime objects once after loading for efficiency
    data["certificates"]["expiry_date"] = pd.to_datetime(data["certificates"]["expiry_date"]).dt.date
    
    return data

def preprocess_data(data):
    """
    Processes raw data to create model-ready parameters, focusing on trainset eligibility.
    """
    train_ids = data["trainsets"]["trainset_id"].tolist()
    eligibility = {}
    today = datetime.date.today()

    for train_id in train_ids:
        is_eligible = True
        
        certs = data["certificates"][data["certificates"]["trainset_id"] == train_id]
        if len(certs) < 3 or (certs["expiry_date"] < today).any():
            is_eligible = False
        
        if not is_eligible:
            eligibility[train_id] = False
            continue

        jobs = data["job_cards"][data["job_cards"]["trainset_id"] == train_id]
        critical_open_jobs = jobs[(jobs["status"] == "OPEN") & (jobs["is_critical"] == True)]
        if not critical_open_jobs.empty:
            is_eligible = False

        eligibility[train_id] = is_eligible
        
    return eligibility

def preprocess_shunting_costs(layout_df):
    """Calculates average costs for key shunting moves."""
    to_maintenance_moves = layout_df[layout_df['to_location'].str.contains('IBL_Bay', na=False)]
    avg_cost_to_maintenance = to_maintenance_moves['shunting_cost'].mean() if not to_maintenance_moves.empty else 0

    to_stabling_moves = layout_df[layout_df['to_location'].str.contains('Stabling_Track', na=False)]
    avg_cost_to_stabling = to_stabling_moves['shunting_cost'].mean() if not to_stabling_moves.empty else 0

    return {
        "maintenance": int(avg_cost_to_maintenance),
        "stabling": int(avg_cost_to_stabling)
    }

def create_and_solve_model(data, eligibility, shunting_costs):
    """
    Builds and solves the model, then RETURNS the solution and key metrics as a DataFrame.
    """
    model = cp_model.CpModel()
    train_ids = data["trainsets"]["trainset_id"].tolist()
    avg_mileage = data["trainsets"]["cumulative_mileage_km"].mean()

    # --- 1. Create Decision Variables ---
    assignments = {}
    for train_id in train_ids:
        assignments[train_id] = {
            "service": model.NewBoolVar(f"{train_id}_service"),
            "standby": model.NewBoolVar(f"{train_id}_standby"),
            "maintenance": model.NewBoolVar(f"{train_id}_maintenance"),
        }

    # --- 2. Add Fundamental Constraints ---
    for train_id in train_ids:
        model.AddExactlyOne(assignments[train_id].values())

    # --- 3. Add Hard Constraints ---
    for train_id, is_eligible in eligibility.items():
        if not is_eligible:
            model.Add(assignments[train_id]["service"] == 0)

    ibl_bays_capacity = data["resources"][data["resources"]["resource_id"] == "IBL_Bays"]["available_capacity"].iloc[0]
    maintenance_trains = [assignments[t]["maintenance"] for t in train_ids]
    model.Add(sum(maintenance_trains) <= ibl_bays_capacity)

    manpower_capacity = data["resources"][data["resources"]["resource_id"] == "Cleaning_Staff_ManHours"]["available_capacity"].iloc[0]
    
    required_hours = {}
    for train_id in train_ids:
        open_jobs = data["job_cards"][(data["job_cards"]["trainset_id"] == train_id) & (data["job_cards"]["status"] == "OPEN")]
        required_hours[train_id] = open_jobs["required_man_hours"].sum()

    total_scheduled_hours = sum(assignments[t]["maintenance"] * int(required_hours.get(t, 0)) for t in train_ids)
    model.Add(total_scheduled_hours <= manpower_capacity)

    # --- 4. Define Objective Function Components ---
    mileage_deviations = []
    for _, train in data["trainsets"].iterrows():
        train_id = train["trainset_id"]
        if train["cumulative_mileage_km"] > avg_mileage:
            deviation = int(train["cumulative_mileage_km"] - avg_mileage)
            mileage_deviations.append(assignments[train_id]["service"] * deviation)
    total_mileage_cost = sum(mileage_deviations)

    sla_penalties = []
    for _, sla in data["slas"].iterrows():
        train_id = sla["trainset_id"]
        if train_id in assignments:
            penalty = int(sla["penalty_per_hour"])
            sla_penalties.append((1 - assignments[train_id]["service"]) * penalty)
    total_branding_penalty = sum(sla_penalties)

    shunting_costs_list = []
    for train_id in train_ids:
        shunting_costs_list.append(assignments[train_id]["maintenance"] * shunting_costs["maintenance"])
        shunting_costs_list.append(assignments[train_id]["service"] * shunting_costs["stabling"])
        shunting_costs_list.append(assignments[train_id]["standby"] * shunting_costs["stabling"])
    total_shunting_cost = sum(shunting_costs_list)

    # --- 5. Set the Final Weighted Objective ---
    w_mileage = 1
    w_branding = 1000
    w_shunting = 10

    model.Minimize(
        w_mileage * total_mileage_cost + 
        w_branding * total_branding_penalty + 
        w_shunting * total_shunting_cost
    )

    # --- 6. Solve the Model ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.solve(model)

    # --- 7. Process and RETURN Results ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution_list = []
        for train_id in train_ids:
            reason = []
            status_str = "Unknown"
            if solver.Value(assignments[train_id]["service"]) == 1: status_str = "Revenue Service"
            elif solver.Value(assignments[train_id]["standby"]) == 1: status_str = "Standby"
            elif solver.Value(assignments[train_id]["maintenance"]) == 1: status_str = "Maintenance"

            if eligibility[train_id]: reason.append("Eligible for service")
            else: reason.append("INELIGIBLE (Cert/Job Card issue)")

            train_mileage = data["trainsets"].loc[data["trainsets"]["trainset_id"] == train_id, "cumulative_mileage_km"].iloc[0]
            mileage_percent_dev = ((train_mileage - avg_mileage) / avg_mileage) * 100
            reason.append(f"Mileage: {train_mileage:,}km ({mileage_percent_dev:+.1f}%)")

            sla_info = data["slas"][data["slas"]["trainset_id"] == train_id]
            if not sla_info.empty:
                current = sla_info['current_exposure_hours'].iloc[0]
                target = sla_info['target_exposure_hours'].iloc[0]
                if current < target:
                    reason.append(f"Branding SLA: ACTIVE ({current}/{target} hrs)")

            hours = required_hours.get(train_id, 0)
            if hours > 0:
                reason.append(f"Pending Work: {int(hours)}h")

            solution_list.append({
                "Trainset ID": train_id,
                "Assigned Status": status_str,
                "Detailed Reasoning": " | ".join(reason)
            })
        
        return pd.DataFrame(solution_list)
    else:
        return None

# This block allows the script to be run standalone for testing
if __name__ == "__main__":
    SCENARIO_FOLDER = 'bottleneck_case'
    print(f"--- Running Solver Standalone for Scenario: {SCENARIO_FOLDER} ---\n")
    
    data_frames = load_data(SCENARIO_FOLDER)
    if data_frames:
        eligibility_dict = preprocess_data(data_frames)
        shunting_costs_dict = preprocess_shunting_costs(data_frames["layout_costs"])
        
        # Capture the returned DataFrame
        solution_df = create_and_solve_model(data_frames, eligibility_dict, shunting_costs_dict)
        
        # Print the DataFrame to the console if a solution was found
        if solution_df is not None:
            print("--- Optimal Train Induction Plan ---")
            print(solution_df.to_string())
        else:
            print("No solution found for the given constraints.")