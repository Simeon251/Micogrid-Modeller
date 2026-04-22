# Microgrid Modeler

Microgrid Modeler is a Streamlit-based design and analysis tool for hybrid microgrids. It lets you enter component datasheet assumptions, optionally load measured resource and demand CSVs, run a time-series simulation, and review technical and financial results through charts, KPI cards, tables, and downloadable outputs.

The current model supports:
- Solar PV
- Wind
- Hydropower
- Diesel generation
- Battery storage
- Demand-side management inputs
- Lifecycle project economics and finance metrics
- Monte Carlo financial risk summaries

## What The App Does

The app in `app.py` provides a user interface for:
- Entering system sizing and datasheet-style assumptions
- Setting site latitude and longitude for synthetic resource generation
- Uploading or referencing resource and load CSV files
- Running the dispatch simulation with `load_following` or `cycle_charging`
- Reviewing reliability, renewable penetration, fuel use, battery behavior, and economics
- Downloading detailed results, metrics, cash flow, and Monte Carlo outputs

The simulation engine lives in `microgrid_simulation.py`.

## Main Files

- `app.py`: Streamlit interface and result visualization
- `microgrid_simulation.py`: time-series simulation, dispatch integration, data ingestion, and lifecycle economics
- `dispatch_model.py`: dispatch logic
- `energy_components.py`: PV, wind, hydro, and diesel component models
- `battery_module.py`: battery storage behavior and degradation
- `demand_model.py`: load profile generation
- `solar_resource_model.py`: synthetic solar resource generation
- `solar_pv_model.py`: PV irradiance transposition and power calculations
- `Example csv/load_profile_example.csv`: sample demand input
- `Example csv/resource_profile_example.csv`: sample resource input

## Core Features

- Flexible simulation timestep support through the engine
- Synthetic resource generation using project latitude and longitude
- Optional measured resource and load CSV ingestion
- Solar PV performance based on irradiance, temperature, tilt, azimuth, and system losses
- Wind, hydro, diesel, and battery dispatch over the model horizon
- Diesel reliability and maintenance modeling
- Demand-side management settings for peak reduction and load shifting
- Lifecycle economics including CAPEX, O&M, replacements, salvage value, and discounted cash flow
- Financial outputs including LCOE, DSCR, and Monte Carlo P10/P50/P90-style summaries

## Supported Inputs

### Site And System Inputs

The app allows you to configure:
- Latitude and longitude
- PV, wind, hydro, diesel, and battery capacities
- Diesel rating in `kW` or `kVA` with power factor treatment in the engine
- PV performance assumptions such as tilt, azimuth, albedo, losses, inverter efficiency, and temperature effects
- Load assumptions and load type
- Economic assumptions such as project life, discount rate, tariff, CAPEX, O&M, debt, and fuel-price escalation

### Optional CSV Inputs

You can provide timestamped CSV inputs instead of relying only on synthetic profiles.

Supported resource/load columns include:
- `timestamp` or `datetime`
- `ghi_w_m2` or `ghi`
- `wind_speed_ms`
- `temperature_c`
- `hydro_flow_m3s`
- `hydro_head_m`
- `load_kw`

If a supported column is missing, the simulator falls back to its internal model for that variable when possible.

Example files are included in the `Example csv` folder.

## Outputs

The app reports both technical and financial results, including:
- Load served fraction
- Renewable fraction
- Fuel consumption
- Diesel runtime and outage metrics
- Battery state of charge behavior
- Load shedding and loss-of-load metrics
- PV specific yield and solar capacity factor
- Upfront CAPEX
- Discounted lifecycle cost
- Operating cost per kWh served
- LCOE
- DSCR metrics
- Monte Carlo summary statistics for finance outputs

Downloadable files from the app include:
- Results time-series CSV
- Metrics CSV
- Lifecycle cash flow CSV
- Monte Carlo summary CSV
- Monte Carlo samples CSV

## Running The App

Install the Python dependencies used by the codebase, then start Streamlit with:

```bash
streamlit run app.py
```

## Running The Simulation Script

You can also run the engine directly:

```bash
python microgrid_simulation.py
```

This runs the built-in example case defined at the bottom of the simulation module.

## Suggested Workflow

1. Enter component datasheet values and operating assumptions in the sidebar.
2. Upload or reference measured resource and load CSVs if available.
3. Run the simulation.
4. Compare reliability, renewable contribution, battery behavior, diesel dependence, and lifecycle cost.
5. Export the outputs for reporting or scenario comparison.

## Notes On Model Use

- Measured resource and demand data should be preferred whenever available.
- Synthetic profiles are most appropriate for early-stage screening rather than bankable studies.
- Scenario analysis is recommended for capacity sizing, fuel sensitivity, tariff sensitivity, and storage tradeoffs.
- Financial outputs depend strongly on the economic assumptions entered by the user.

## Status

This project is structured so additional technologies and modeling improvements can be added over time.
