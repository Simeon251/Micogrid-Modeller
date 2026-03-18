
import pandas as pd
from energy_components import DieselGenerator, WindTurbine, PVGenerator
from battery_module import LeadAcidBattery, KiBaMBattery

def load_following_algorithm(load_kw,
                            solar_power_kw,
                            wind_power_kw,
                            battery,
                            diesel_generator,
                            battery_operating_cost=0.1,
                            timestep_hr=1.0,
                            min_soc=0.2,
                            dispatch_strategy='load_following'):
    """Dispatch algorithm (load following) for a single timestep.

    Priorities:
      1) Solar + wind (renewables)
      2) Battery discharge (if deficit) or diesel (as backup)
      3) Battery charge (with excess generation)
      4) Curtailment or load shedding if resources insufficient

    The algorithm tracks energy flows (kWh) to ensure energy balance
    within each timestep.
    """

    dispatch = {
        'solar': 0.0,
        'wind': 0.0,
        'battery': 0.0,  # positive: discharge, negative: charge
        'diesel': 0.0,
        'battery_discharge_kwh': 0.0,
        'battery_charge_kwh': 0.0,
        'load_served_kw': 0.0,
        'load_shedding_kw': 0.0,
        'curtailment': 0.0,
        'operating_cost': 0.0,
        'battery_soc_before': battery.get_soc() * 100,
        'fuel_liters': 0.0,
        'error_kw': 0.0,
    }

    # Convert to energy for timestep (kWh)
    load_kwh = load_kw * timestep_hr
    solar_kwh = solar_power_kw * timestep_hr
    wind_kwh = wind_power_kw * timestep_hr

    # Track renewable contribution
    dispatch['solar'] = solar_power_kw
    dispatch['wind'] = wind_power_kw

    # Start with renewable energy
    supply_kwh = solar_kwh + wind_kwh

    # 1) Use battery discharge to meet deficit before resorting to diesel
    deficit_kwh = max(0.0, load_kwh - supply_kwh)
    if deficit_kwh > 0:
        max_discharge_kw = battery.get_max_discharge_power(timestep_hr=timestep_hr)
        discharge_kwh = min(max_discharge_kw * timestep_hr, deficit_kwh)
        if discharge_kwh > 0:
            discharge_kw = discharge_kwh / timestep_hr
            actual_discharge_kwh = battery.discharge(discharge_kw, timestep_hr=timestep_hr)
            dispatch['battery_discharge_kwh'] = actual_discharge_kwh
            dispatch['battery'] = actual_discharge_kwh / timestep_hr
            dispatch['operating_cost'] += actual_discharge_kwh * battery_operating_cost
            supply_kwh += actual_discharge_kwh
            deficit_kwh = max(0.0, deficit_kwh - actual_discharge_kwh)

    # 2) Use diesel generator to cover remaining deficit
    if deficit_kwh > 1e-6:
        if dispatch_strategy == 'cycle_charging':
            # Run generator at rated power when started, then absorb any excess in the battery.
            diesel_kw = diesel_generator.rated_kw
        else:
            diesel_kw = min(diesel_generator.rated_kw, deficit_kwh / timestep_hr)

        if diesel_kw > 0 and diesel_kw < diesel_generator.min_load_factor * diesel_generator.rated_kw:
            diesel_kw = diesel_generator.min_load_factor * diesel_generator.rated_kw

        diesel_kwh = diesel_kw * timestep_hr
        dispatch['diesel'] = diesel_kw
        supply_kwh += diesel_kwh

        fuel_liters = diesel_generator.fuel_consumption(diesel_kw, timestep_hr)
        dispatch['fuel_liters'] = fuel_liters
        dispatch['operating_cost'] += fuel_liters * 1.50

        # Update deficit (if diesel is running below demand, it creates surplus)
        deficit_kwh = max(0.0, deficit_kwh - diesel_kwh)

    # 3) Determine load served and shedding
    load_served_kwh = min(load_kwh, supply_kwh)
    load_shedding_kwh = max(0.0, load_kwh - supply_kwh)
    dispatch['load_served_kw'] = load_served_kwh / timestep_hr
    dispatch['load_shedding_kw'] = load_shedding_kwh / timestep_hr

    # 4) Use any remaining supply to charge the battery (if possible)
    supply_after_load_kwh = max(0.0, supply_kwh - load_served_kwh)
    if supply_after_load_kwh > 1e-6:
        max_charge_kw = battery.get_max_charge_power(timestep_hr=timestep_hr)
        charge_kwh_desired = min(max_charge_kw * timestep_hr, supply_after_load_kwh)
        if charge_kwh_desired > 0:
            charge_kw = charge_kwh_desired / timestep_hr
            drawn_kwh = battery.charge(charge_kw, timestep_hr=timestep_hr)
            dispatch['battery_charge_kwh'] = drawn_kwh
            dispatch['battery'] = -(drawn_kwh / timestep_hr)
            supply_after_load_kwh = max(0.0, supply_after_load_kwh - drawn_kwh)

    # 5) Any remaining supply is curtailed
    dispatch['curtailment'] = supply_after_load_kwh / timestep_hr

    # Energy balance check (should be near zero)
    energy_used_by_load = load_served_kwh
    energy_to_battery = dispatch['battery_charge_kwh']
    energy_curtailed = dispatch['curtailment'] * timestep_hr
    imbalance_kwh = supply_kwh - (energy_used_by_load + energy_to_battery + energy_curtailed)
    dispatch['error_kw'] = imbalance_kwh / max(timestep_hr, 1e-6)

    dispatch['battery_soc_after'] = battery.get_soc() * 100

    return dispatch


def run_dispatch_simulation(df, pv_gen, wind_turbine, battery, diesel_generator, battery_operating_cost=0.1):

    results = []
    
    for idx, row in df.iterrows():
        # Calculate solar power
        solar_power = pv_gen.ac_power(row['G W/sqm'], row['Ta ˚C'], year=0)
        
        # Calculate wind power from wind speed at 20m
        wind_speed_at_hub = wind_turbine.wind_speed_at_hub(row['Wind at 20m (m/s)'], ref_height=20.0)
        wind_power = wind_turbine.power_output(wind_speed_at_hub, row['Ta ˚C'])
        
        # Get load
        load_kw = row['Load (kW)']
        
        dispatch = load_following_algorithm(
            load_kw=load_kw,
            solar_power_kw=solar_power,
            wind_power_kw=wind_power,
            battery=battery,
        diesel_generator=diesel_generator,
        battery_operating_cost=battery_operating_cost,
        timestep_hr=1/12.0,  # Data is every 15 minutes
        dispatch_strategy='load_following'
        )
        dispatch['timestep'] = idx
        dispatch['solar_available'] = solar_power
        dispatch['wind_available'] = wind_power
        dispatch['load'] = load_kw
        results.append(dispatch)
    
    return pd.DataFrame(results)


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    # Create battery and generator instances
    battery = LeadAcidBattery(
        energy_capacity_kwh=500.0,
        power_capacity_kw=30.0,
        charge_efficiency=0.93,
        discharge_efficiency=0.93,
        max_depth_of_discharge=0.80
    )
    
    diesel_gen = DieselGenerator(
        standby_kw=52.8,
        prime_kw=48.0,
        fuel_curve_lph={0.25: 5.0, 0.50: 8.2, 0.75: 12.2, 1.0: 16.1},
        min_load_factor=0.25
    )
    
    # Single dispatch example
    print("=" * 60)
    print("SINGLE TIMESTEP DISPATCH EXAMPLE")
    print("=" * 60)
    load = 125.0  # kW
    solar = 20.0  # kW
    wind = 25.0  # kW
    
    dispatch = load_following_algorithm(
        load_kw=load,
        solar_power_kw=solar,
        wind_power_kw=wind,
        battery=battery,
        diesel_generator=diesel_gen,
        battery_operating_cost=0.1,
        timestep_hr=1.0
    )
    
    print(f"Load: {load} kW")
    print(f"Solar: {solar} kW")
    print(f"Wind: {wind} kW")
    print("\nDispatch Results:")
    for key, value in dispatch.items():
        if 'soc' in key.lower():
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.3f}")
    





