"""Dispatch logic for a single microgrid timestep."""


def load_following_algorithm(load_kw,
                            solar_power_kw,
                            wind_power_kw,
                            hydro_power_kw,
                            battery,
                            diesel_generator,
                            battery_variable_cost_per_kwh=0.0,
                            diesel_fuel_price_per_liter=1.50,
                            diesel_variable_om_per_kwh=0.0,
                            timestep_hr=1.0,
                            dispatch_strategy='load_following'):
    """Dispatch a single timestep using renewable-first load following."""

    dispatch = {
        'solar': 0.0,
        'wind': 0.0,
        'hydro': 0.0,
        'battery': 0.0,  # positive: discharge, negative: charge
        'diesel': 0.0,
        'battery_discharge_kwh': 0.0,
        'battery_charge_kwh': 0.0,
        'load_served_kw': 0.0,
        'load_shedding_kw': 0.0,
        'renewable_served_kwh': 0.0,
        'battery_served_kwh': 0.0,
        'diesel_served_kwh': 0.0,
        'curtailment': 0.0,
        'operating_cost': 0.0,
        'battery_soc_before': battery.get_soc() * 100,
        'fuel_liters': 0.0,
        'error_kw': 0.0,
    }

    # Convert power to timestep energy for balance calculations.
    load_kwh = load_kw * timestep_hr
    solar_kwh = solar_power_kw * timestep_hr
    wind_kwh = wind_power_kw * timestep_hr
    hydro_kwh = hydro_power_kw * timestep_hr
    renewable_kwh = solar_kwh + wind_kwh + hydro_kwh

    dispatch['solar'] = solar_power_kw
    dispatch['wind'] = wind_power_kw
    dispatch['hydro'] = hydro_power_kw

    supply_kwh = renewable_kwh

    # Use the battery before diesel when renewables cannot fully meet demand.
    deficit_kwh = max(0.0, load_kwh - supply_kwh)
    if deficit_kwh > 0:
        max_discharge_kw = battery.get_max_discharge_power(timestep_hr=timestep_hr)
        discharge_kwh = min(max_discharge_kw * timestep_hr, deficit_kwh)
        if discharge_kwh > 0:
            discharge_kw = discharge_kwh / timestep_hr
            actual_discharge_kwh = battery.discharge(discharge_kw, timestep_hr=timestep_hr)
            dispatch['battery_discharge_kwh'] = actual_discharge_kwh
            dispatch['battery'] = actual_discharge_kwh / timestep_hr
            dispatch['operating_cost'] += actual_discharge_kwh * battery_variable_cost_per_kwh
            supply_kwh += actual_discharge_kwh
            deficit_kwh = max(0.0, deficit_kwh - actual_discharge_kwh)

    if deficit_kwh > 1e-6:
        if dispatch_strategy == 'cycle_charging':
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
        dispatch['operating_cost'] += fuel_liters * diesel_fuel_price_per_liter
        dispatch['operating_cost'] += diesel_kwh * diesel_variable_om_per_kwh

    load_served_kwh = min(load_kwh, supply_kwh)
    load_shedding_kwh = max(0.0, load_kwh - supply_kwh)
    dispatch['load_served_kw'] = load_served_kwh / timestep_hr
    dispatch['load_shedding_kw'] = load_shedding_kwh / timestep_hr
    dispatch['renewable_served_kwh'] = min(load_kwh, renewable_kwh)
    remaining_after_renewables = max(0.0, load_served_kwh - dispatch['renewable_served_kwh'])
    dispatch['battery_served_kwh'] = min(remaining_after_renewables, dispatch['battery_discharge_kwh'])
    dispatch['diesel_served_kwh'] = max(
        0.0,
        load_served_kwh - dispatch['renewable_served_kwh'] - dispatch['battery_served_kwh']
    )

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

    dispatch['curtailment'] = supply_after_load_kwh / timestep_hr

    energy_used_by_load = load_served_kwh
    energy_to_battery = dispatch['battery_charge_kwh']
    energy_curtailed = dispatch['curtailment'] * timestep_hr
    imbalance_kwh = supply_kwh - (energy_used_by_load + energy_to_battery + energy_curtailed)
    dispatch['error_kw'] = imbalance_kwh / max(timestep_hr, 1e-6)
    dispatch['battery_soc_after'] = battery.get_soc() * 100

    return dispatch
