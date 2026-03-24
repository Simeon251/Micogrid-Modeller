"""Dispatch logic for a single microgrid timestep."""


def _economic_dispatch_solution(load_kw,
                                renewable_kwh,
                                battery,
                                diesel_generator,
                                diesel_available,
                                battery_variable_cost_per_kwh,
                                diesel_fuel_price_per_liter,
                                diesel_variable_om_per_kwh,
                                diesel_maintenance_cost_per_hour,
                                ambient_temp_c,
                                timestep_hr):
    """Solve a single-step economic dispatch by enumerating feasible diesel setpoints."""
    load_kwh = load_kw * timestep_hr
    max_charge_kw = battery.get_max_charge_power(timestep_hr=timestep_hr)
    max_discharge_kw = battery.get_max_discharge_power(timestep_hr=timestep_hr)
    max_charge_kwh = max_charge_kw * timestep_hr
    max_discharge_kwh = max_discharge_kw * timestep_hr

    deficit_kwh = load_kwh - renewable_kwh
    rated_kw = diesel_generator.rated_kw if diesel_available else 0.0
    min_load_kw = diesel_generator.min_load_factor * rated_kw if rated_kw > 0 else 0.0

    candidate_diesel_kw = {
        0.0,
        rated_kw,
        max(0.0, deficit_kwh / timestep_hr) if timestep_hr > 0 else 0.0,
        min_load_kw if deficit_kwh > 0 else 0.0,
    }

    feasible = []
    for diesel_kw in sorted(candidate_diesel_kw):
        if diesel_kw < 0 or diesel_kw > rated_kw + 1e-9:
            continue
        if 0 < diesel_kw < min_load_kw:
            continue

        diesel_kwh = diesel_kw * timestep_hr
        net_after_generation = renewable_kwh + diesel_kwh - load_kwh

        battery_charge_kwh = 0.0
        battery_discharge_kwh = 0.0
        curtailment_kwh = 0.0
        load_shedding_kwh = 0.0

        if net_after_generation >= 0:
            battery_charge_kwh = min(net_after_generation, max_charge_kwh)
            curtailment_kwh = max(0.0, net_after_generation - battery_charge_kwh)
        else:
            battery_discharge_kwh = min(-net_after_generation, max_discharge_kwh)
            load_shedding_kwh = max(0.0, -net_after_generation - battery_discharge_kwh)

        if diesel_kw > 0:
            fuel_liters = diesel_generator.estimate_fuel_consumption(diesel_kw, timestep_hr)
            diesel_cost = (
                fuel_liters * diesel_fuel_price_per_liter +
                diesel_kwh * diesel_variable_om_per_kwh +
                diesel_maintenance_cost_per_hour * timestep_hr
            )
        else:
            fuel_liters = 0.0
            diesel_cost = 0.0

        battery_cost = battery_variable_cost_per_kwh * battery_discharge_kwh
        total_cost = diesel_cost + battery_cost

        feasible.append({
            'diesel_kw': diesel_kw,
            'diesel_kwh': diesel_kwh,
            'fuel_liters': fuel_liters,
            'battery_charge_kwh': battery_charge_kwh,
            'battery_discharge_kwh': battery_discharge_kwh,
            'curtailment_kwh': curtailment_kwh,
            'load_shedding_kwh': load_shedding_kwh,
            'total_cost': total_cost,
        })

    feasible.sort(key=lambda item: (item['load_shedding_kwh'] > 1e-6, item['total_cost'], item['diesel_kw']))
    return feasible[0] if feasible else None


def load_following_algorithm(load_kw,
                            solar_power_kw,
                            wind_power_kw,
                            hydro_power_kw,
                            battery,
                            diesel_generator,
                            diesel_available=True,
                            battery_variable_cost_per_kwh=0.0,
                            diesel_fuel_price_per_liter=1.50,
                            diesel_variable_om_per_kwh=0.0,
                            diesel_maintenance_cost_per_hour=0.0,
                            ambient_temp_c=None,
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

    if dispatch_strategy == 'economic_dispatch':
        solution = _economic_dispatch_solution(
            load_kw=load_kw,
            renewable_kwh=renewable_kwh,
            battery=battery,
            diesel_generator=diesel_generator,
            diesel_available=diesel_available,
            battery_variable_cost_per_kwh=battery_variable_cost_per_kwh,
            diesel_fuel_price_per_liter=diesel_fuel_price_per_liter,
            diesel_variable_om_per_kwh=diesel_variable_om_per_kwh,
            diesel_maintenance_cost_per_hour=diesel_maintenance_cost_per_hour,
            ambient_temp_c=ambient_temp_c,
            timestep_hr=timestep_hr,
        )
        if solution is not None:
            if solution['battery_discharge_kwh'] > 0:
                discharge_kw = solution['battery_discharge_kwh'] / timestep_hr
                actual_discharge_kwh = battery.discharge(
                    discharge_kw,
                    timestep_hr=timestep_hr,
                    temp_c=ambient_temp_c
                )
                dispatch['battery_discharge_kwh'] = actual_discharge_kwh
                dispatch['battery'] = actual_discharge_kwh / timestep_hr
                dispatch['battery_served_kwh'] = actual_discharge_kwh
                dispatch['operating_cost'] += actual_discharge_kwh * battery_variable_cost_per_kwh

            if solution['diesel_kw'] > 0:
                dispatch['diesel'] = solution['diesel_kw']
                dispatch['fuel_liters'] = diesel_generator.fuel_consumption(solution['diesel_kw'], timestep_hr)
                dispatch['operating_cost'] += dispatch['fuel_liters'] * diesel_fuel_price_per_liter
                dispatch['operating_cost'] += solution['diesel_kwh'] * diesel_variable_om_per_kwh
                dispatch['operating_cost'] += diesel_maintenance_cost_per_hour * timestep_hr

            if solution['battery_charge_kwh'] > 0:
                charge_kw = solution['battery_charge_kwh'] / timestep_hr
                drawn_kwh = battery.charge(
                    charge_kw,
                    timestep_hr=timestep_hr,
                    temp_c=ambient_temp_c
                )
                dispatch['battery_charge_kwh'] = drawn_kwh
                dispatch['battery'] = -(drawn_kwh / timestep_hr)

            supply_kwh = renewable_kwh + dispatch['diesel'] * timestep_hr + dispatch['battery_discharge_kwh']
            load_served_kwh = min(load_kwh, supply_kwh)
            dispatch['load_served_kw'] = load_served_kwh / timestep_hr
            dispatch['load_shedding_kw'] = max(0.0, load_kwh - supply_kwh) / timestep_hr
            dispatch['renewable_served_kwh'] = min(load_kwh, renewable_kwh)
            remaining_after_renewables = max(0.0, load_served_kwh - dispatch['renewable_served_kwh'])
            dispatch['battery_served_kwh'] = min(remaining_after_renewables, dispatch['battery_discharge_kwh'])
            dispatch['diesel_served_kwh'] = max(
                0.0,
                load_served_kwh - dispatch['renewable_served_kwh'] - dispatch['battery_served_kwh']
            )
            dispatch['curtailment'] = max(
                0.0,
                renewable_kwh + dispatch['diesel'] * timestep_hr -
                load_served_kwh - dispatch['battery_charge_kwh']
            ) / timestep_hr
            imbalance_kwh = (
                renewable_kwh + dispatch['diesel'] * timestep_hr + dispatch['battery_discharge_kwh'] -
                (load_served_kwh + dispatch['battery_charge_kwh'] + dispatch['curtailment'] * timestep_hr)
            )
            dispatch['error_kw'] = imbalance_kwh / max(timestep_hr, 1e-6)
            dispatch['battery_soc_after'] = battery.get_soc() * 100
            return dispatch

    supply_kwh = renewable_kwh

    # Use the battery before diesel when renewables cannot fully meet demand.
    deficit_kwh = max(0.0, load_kwh - supply_kwh)
    if deficit_kwh > 0:
        max_discharge_kw = battery.get_max_discharge_power(timestep_hr=timestep_hr)
        discharge_kwh = min(max_discharge_kw * timestep_hr, deficit_kwh)
        if discharge_kwh > 0:
            discharge_kw = discharge_kwh / timestep_hr
            actual_discharge_kwh = battery.discharge(
                discharge_kw,
                timestep_hr=timestep_hr,
                temp_c=ambient_temp_c
            )
            dispatch['battery_discharge_kwh'] = actual_discharge_kwh
            dispatch['battery'] = actual_discharge_kwh / timestep_hr
            dispatch['operating_cost'] += actual_discharge_kwh * battery_variable_cost_per_kwh
            supply_kwh += actual_discharge_kwh
            deficit_kwh = max(0.0, deficit_kwh - actual_discharge_kwh)

    rated_kw = diesel_generator.rated_kw if diesel_available else 0.0
    min_load_kw = diesel_generator.min_load_factor * rated_kw if rated_kw > 0 else 0.0

    if deficit_kwh > 1e-6 and rated_kw > 0:
        if dispatch_strategy == 'cycle_charging':
            diesel_kw = rated_kw
        else:
            diesel_kw = min(rated_kw, deficit_kwh / timestep_hr)

        if diesel_kw > 0 and diesel_kw < min_load_kw:
            diesel_kw = min_load_kw

        diesel_kwh = diesel_kw * timestep_hr
        dispatch['diesel'] = diesel_kw
        supply_kwh += diesel_kwh

        fuel_liters = diesel_generator.fuel_consumption(diesel_kw, timestep_hr)
        dispatch['fuel_liters'] = fuel_liters
        dispatch['operating_cost'] += fuel_liters * diesel_fuel_price_per_liter
        dispatch['operating_cost'] += diesel_kwh * diesel_variable_om_per_kwh
        dispatch['operating_cost'] += diesel_maintenance_cost_per_hour * timestep_hr

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
    # Avoid charging and discharging in the same timestep.
    if supply_after_load_kwh > 1e-6 and dispatch['battery_discharge_kwh'] <= 1e-9:
        max_charge_kw = battery.get_max_charge_power(timestep_hr=timestep_hr)
        charge_kwh_desired = min(max_charge_kw * timestep_hr, supply_after_load_kwh)
        if charge_kwh_desired > 0:
            charge_kw = charge_kwh_desired / timestep_hr
            drawn_kwh = battery.charge(
                charge_kw,
                timestep_hr=timestep_hr,
                temp_c=ambient_temp_c
            )
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
