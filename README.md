# MicroGridModeler

MicroGridModeler is a user-facing microgrid design tool. A user can enter datasheet values for the currently implemented sources and storage systems, run the simulation, and review the output through clearer visuals and downloadable result tables. The simulator now includes lifecycle project economics with CAPEX, O&M, replacement scheduling, fuel-price escalation, discounted cash flow, and LCOE reporting. It also supports optional timestamped meteorological/resource CSV inputs and a hydropower plant model. The codebase is organized so other energy sources can be added later.

## Main Files
- `streamlit_app.py` - Streamlit interface for entering datasheet values and viewing outputs
- `microgrid_simulation.py` - main simulation engine, resource-data ingestion, and lifecycle economics model
- `dispatch_model.py` - dispatch optimization logic
- `energy_components.py` - component models for the currently implemented energy sources, including hydropower
- `battery_module.py` - battery system behavior
- `demand_model.py` - load profile generation
- `solar_resource_model.py` - solar resource generation
- `solar_pv_model.py` - PV performance calculations

## Economic Outputs
- Upfront CAPEX
- Operating cost per kWh served
- Lifecycle cost (NPV)
- LCOE
- Replacement schedule cash flow
- Fuel use and outage-penalty costs

## Resource CSV Inputs
You can optionally point the app to a timestamped resource CSV. Supported column names include:
- `timestamp` or `datetime`
- `ghi_w_m2` or `ghi`
- `wind_speed_ms`
- `temperature_c`
- `hydro_flow_m3s`
- `hydro_head_m`
- `load_kw`

If one of these columns is missing, the simulator falls back to its internal synthetic model for that variable.

## Run The Streamlit App
```bash
streamlit run streamlit_app.py
```

## Run The Script Version
```bash
python microgrid_simulation.py
```
