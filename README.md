# MicroGridModeler

MicroGridModeler is a user-facing microgrid design tool. A user can enter datasheet values for the currently implemented sources and storage systems, run the simulation, and review the output through clearer visuals and downloadable result tables. The codebase is organized so other energy sources can be added later.

## Main Files
- `streamlit_app.py` - Streamlit interface for entering datasheet values and viewing outputs
- `microgrid_simulation.py` - main simulation engine
- `dispatch_model.py` - dispatch optimization logic
- `energy_components.py` - component models for the currently implemented energy sources
- `battery_module.py` - battery system behavior
- `demand_model.py` - load profile generation
- `solar_resource_model.py` - solar resource generation
- `solar_pv_model.py` - PV performance calculations

## Run The Streamlit App
```bash
streamlit run streamlit_app.py
```

## Run The Script Version
```bash
python microgrid_simulation.py
```
