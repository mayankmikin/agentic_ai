import json
from langchain_core.tools import tool

# 2. Domain Data & Custom Tools for Scandinavian Fisheries
FISHERIES_DATABASE = {
    "atlantic_cod": {
        "common_name": "Atlantic Cod",
        "region": "Barents Sea / Norwegian Sea (ICES 1-2)",
        "stock_status": "Healthy / Below F_msy",
        "tac_tonnes": 453000,
        "reported_catch_tonnes": 380000,
    },
    "north_sea_herring": {
        "common_name": "North Sea Herring",
        "region": "North Sea, Skagerrak, Kattegat (ICES 4, 3a)",
        "stock_status": "Sustainable",
        "tac_tonnes": 532000,
        "reported_catch_tonnes": 510000,
    },
    "atlantic_salmon": {
        "common_name": "Atlantic Salmon (Aquaculture)",
        "region": "Norwegian Coastal Aquaculture (Traffic Light Zone 3)",
        "stock_status": "Green Zone (Low Sea Lice Impact)",
        "max_allowable_biomass_tonnes": 82000,
        "current_biomass_tonnes": 74500,
        "tac_tonnes": 82000,
        "reported_catch_tonnes": 74500,
    }
}

@tool
def get_fish_stock_data(species_key: str) -> str:
    """Fetch current TAC quotas, stock health, and catch statistics for a Scandinavian fish species.
    Accepted species_key values: 'atlantic_cod', 'north_sea_herring', 'atlantic_salmon'
    """
    key = species_key.lower().replace(" ", "_")
    data = FISHERIES_DATABASE.get(key)
    if not data:
        return f"Error: No data available for '{species_key}'. Available keys: {list(FISHERIES_DATABASE.keys())}"
    return json.dumps(data, indent=2)

@tool
def evaluate_catch_compliance(species_key: str, requested_tonnes: float) -> str:
    """Evaluate if a proposed commercial catch volume complies with remaining Scandinavian TAC quotas."""
    key = species_key.lower().replace(" ", "_")
    data = FISHERIES_DATABASE.get(key)
    if not data:
        return f"Error: Unknown species '{species_key}'."

    tac = data.get("tac_tonnes") or data.get("max_allowable_biomass_tonnes")
    reported = data.get("reported_catch_tonnes") or data.get("current_biomass_tonnes")
    remaining_quota = tac - reported

    if requested_tonnes <= remaining_quota:
        return (
            f"APPROVED: {requested_tonnes:,.1f} tonnes requested. "
            f"Remaining quota: {remaining_quota:,.1f} tonnes. "
            f"New quota remaining after catch: {(remaining_quota - requested_tonnes):,.1f} tonnes."
        )
    else:
        return (
            f"REJECTED: Requested {requested_tonnes:,.1f} tonnes exceeds the remaining "
            f"safe quota of {remaining_quota:,.1f} tonnes by {(requested_tonnes - remaining_quota):,.1f} tonnes."
        )

# 3. Assemble Tools and Build the LangGraph Agent
available_tools = [get_fish_stock_data, evaluate_catch_compliance]