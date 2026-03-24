"""
Integrated Microgrid System Performance Simulation

This module provides a comprehensive simulation framework for microgrids that:
1. Initializes all components (generators, batteries, loads, resources)
2. Creates flexible time indices (15 min, 30 min, 1 hr timesteps)
3. Generates resource data over the model horizon
4. Generates load profiles
5. Runs dispatch algorithm for each timestep
6. Updates component states
7. Saves comprehensive performance metrics

"""

import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from solar_pv_model import pv_power_from_ghi, extraterrestrial_hour
from battery_module import KiBaMBattery
from demand_model import MicrogridLoad
from dispatch_model import load_following_algorithm
from energy_components import PVGenerator, DieselGenerator, WindTurbine, HydroTurbine
from solar_resource_model import SolarResourceSimulator


class MicrogridSimulation:
    """
    Complete microgrid simulation framework for performance analysis over time.
    
    Supports:
    - Flexible timesteps (15 min, 30 min, 1 hr, customizable)
    - Multiple generator types (PV, Wind, Diesel)
    - Battery storage with degradation
    - Realistic load profiles
    - Stochastic resource variability
    - Comprehensive performance tracking
    """
    
    def __init__(self,
                 timestep_minutes=60,
                 num_days=365,
                 start_date='2026-01-01',
                 location_lat=-1.94,
                 location_lon=30.06,
                 pv_capacity_kwp=500.0,
                 wind_capacity_kw=200.0,
                 hydro_capacity_kw=0.0,
                 diesel_capacity_kw=100.0,
                 diesel_capacity_kva=None,
                 diesel_power_factor=0.8,
                 battery_capacity_kwh=500.0,
                 battery_power_kw=50.0,
                 base_load_kw=50.0,
                 load_type='residential',
                 load_profile_file=None,
                 resource_profile_file=None,
                 dispatch_strategy='load_following',
                 pv_params=None,
                 wind_params=None,
                 hydro_params=None,
                 diesel_params=None,
                 battery_params=None,
                 load_params=None,
                 economic_params=None,
                 random_seed=None):
        """
        Initialize microgrid simulation.
        
        Args:
            timestep_minutes (int): Timestep in minutes (must divide 1440)
            num_days (int): Simulation duration in days
            start_date (str): Start date in 'YYYY-MM-DD' format
            pv_capacity_kwp (float): PV array capacity in kWp
            wind_capacity_kw (float): Wind turbine capacity in kW
            hydro_capacity_kw (float): Hydropower turbine rated capacity in kW
            diesel_capacity_kw (float): Diesel generator capacity in kW
            battery_capacity_kwh (float): Battery energy capacity in kWh
            battery_power_kw (float): Battery power capacity in kW
            base_load_kw (float): Base load demand in kW
            load_type (str): 'residential', 'commercial', 'industrial', or 'custom'
            load_profile_file (str | None): Optional CSV file containing a provided load profile
            resource_profile_file (str | None): Optional CSV containing timestamped meteorological and hydro resource data
            dispatch_strategy (str): 'load_following' or 'cycle_charging'
            pv_params (dict | None): Optional PV datasheet/model overrides
            wind_params (dict | None): Optional wind turbine datasheet/model overrides
            hydro_params (dict | None): Optional hydropower model overrides
            diesel_params (dict | None): Optional diesel generator datasheet/model overrides
            battery_params (dict | None): Optional battery model overrides
            load_params (dict | None): Optional load model overrides
            economic_params (dict | None): Optional lifecycle cost/economic overrides
            random_seed (int | None): Optional random seed for repeatable runs
        """
        
        # Validate timestep
        if 1440 % timestep_minutes != 0:
            raise ValueError(f"timestep_minutes ({timestep_minutes}) must divide 1440 evenly")
        
        self.timestep_minutes = timestep_minutes
        self.timestep_hours = timestep_minutes / 60.0
        self.num_days = num_days
        self.steps_per_day = 1440 // timestep_minutes
        self.total_steps = num_days * self.steps_per_day
        self.dispatch_strategy = dispatch_strategy
        self.load_profile_file = load_profile_file
        self.resource_profile_file = resource_profile_file
        self.random_seed = random_seed
        self.location_lat = float(location_lat)
        self.location_lon = float(location_lon)
        self.simulated_years = max(num_days / 365.0, 1.0 / 365.0)

        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Parse start date
        self.start_date = pd.to_datetime(start_date)
        self.end_date = self.start_date + timedelta(days=num_days)
        
        print("="*70)
        print("MICROGRID SIMULATION INITIALIZATION")
        print("="*70)
        print(f"Timestep: {timestep_minutes} minutes")
        print(f"Simulation period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Duration: {num_days} days ({self.total_steps} timesteps)")
        print(f"Steps per day: {self.steps_per_day}")
        print()
        
        # Initialize components
        print("Initializing microgrid components...")

        # Allow specifying generator rating in kVA with a power factor (e.g. 60 kVA @ 0.8 PF -> 48 kW)
        if diesel_capacity_kva is not None:
            diesel_capacity_kw = diesel_capacity_kva * diesel_power_factor

        pv_model_overrides = {}
        if pv_params:
            pv_model_overrides = {
                key: pv_params[key]
                for key in ['isc_temp_coeff_rel']
                if key in pv_params
            }

        pv_config = {'array_capacity_kwp': pv_capacity_kwp}
        if pv_params:
            pv_config.update({k: v for k, v in pv_params.items() if k not in pv_model_overrides})
        self.pv_gen = PVGenerator(**pv_config)
        self.pv_model_params = {
            'isc_temp_coeff_rel': pv_model_overrides.get('isc_temp_coeff_rel', 0.0005)
        }

        wind_config = {'rated_power_kw': wind_capacity_kw}
        if wind_params:
            wind_config.update(wind_params)
        self.wind_turbine = WindTurbine(**wind_config)

        hydro_config = {'rated_power_kw': hydro_capacity_kw}
        if hydro_params:
            hydro_config.update(hydro_params)
        self.hydro_turbine = HydroTurbine(**hydro_config)

        diesel_config = {
            'standby_kva': diesel_capacity_kva if diesel_capacity_kva is not None else None,
            'prime_kva': diesel_capacity_kva if diesel_capacity_kva is not None else None,
            'standby_kw': diesel_capacity_kw * 1.1,
            'prime_kw': diesel_capacity_kw,
            'power_factor': diesel_power_factor,
            'fuel_curve_lph': {0.25: 4.50, 0.50: 7.40, 0.75: 11.00, 1.0: 14.70},
            'min_load_factor': 0.25
        }
        if diesel_params:
            diesel_config.update({
                k: v for k, v in diesel_params.items()
                if k not in {
                    'enable_generator_reliability',
                    'mtbf_hours',
                    'mttr_hours',
                    'planned_maintenance_interval_hours',
                    'planned_maintenance_duration_hours',
                }
            })
        self.diesel_gen = DieselGenerator(**diesel_config)
        self.diesel_reliability = self._build_diesel_reliability_params(diesel_params)
        self.diesel_outage_remaining_steps = 0
        self.diesel_outage_reason = "available"
        self.next_planned_maintenance_runtime_hours = self.diesel_reliability['planned_maintenance_interval_hours']
        self.diesel_forced_outage_events = 0
        self.diesel_planned_outage_events = 0
        # Use the KiBaM battery model for current-based storage dynamics.
        battery_config = {
            'energy_capacity_kwh': battery_capacity_kwh,
            'power_capacity_kw': battery_power_kw,
            'nominal_voltage': 48.0,
            'k_rate': 0.1,
            'c_fraction': 0.3
        }
        if battery_params:
            battery_config.update(battery_params)
        self.battery = KiBaMBattery(**battery_config)

        load_config = {
            'base_kw': base_load_kw,
            'timestep_minutes': timestep_minutes,
            'load_type': load_type,
            'base_year': self.start_date.year
        }
        if load_params:
            load_config.update({
                k: v for k, v in load_params.items()
                if k not in {
                    'enable_dsm',
                    'deferrable_load_fraction',
                    'peak_reduction_fraction',
                    'peak_start_hour',
                    'peak_end_hour',
                    'shift_start_hour',
                    'shift_end_hour',
                }
            })
        self.load_model = MicrogridLoad(**load_config)
        self.dsm_params = self._build_dsm_params(load_params)
        self.dsm_summary = {}
        self.economic_params = self._build_economic_params(economic_params)
        self._last_battery_throughput_mwh = 0.0

        print(f"  + PV: {pv_capacity_kwp} kWp")
        print(f"  + Wind: {wind_capacity_kw} kW")
        print(f"  + Hydro: {hydro_capacity_kw} kW")
        if diesel_capacity_kva is not None:
            print(f"  + Diesel: {diesel_capacity_kva} kVA @ {diesel_power_factor} PF (~ {diesel_capacity_kw:.1f} kW)")
        else:
            print(f"  + Diesel: {diesel_capacity_kw} kW")
        print(f"  + Battery: {battery_capacity_kwh} kWh / {battery_power_kw} kW")
        print(f"  + Load: {base_load_kw} kW base ({load_type})")
        print(f"  + Dispatch: {dispatch_strategy}")
        print(f"  + Synthetic resource location: ({self.location_lat:.2f}, {self.location_lon:.2f})")
        print(f"  + Project life: {self.economic_params['project_lifetime_years']} years")
        if self.dsm_params['enable_dsm']:
            print("  + DSM enabled")
        if self.diesel_reliability['enable_generator_reliability']:
            print("  + Diesel reliability enabled")
        if load_profile_file:
            print(f"  + Load profile file: {load_profile_file}")
        if resource_profile_file:
            print(f"  + Resource profile file: {resource_profile_file}")
        print()
        
        # Create time index
        self.time_index = self._create_time_index()
        
        # Initialize results storage
        self.results = []
        self.system_states = []
        self.performance_metrics = {}
        self.economic_cashflow = pd.DataFrame()
        self.monte_carlo_summary = pd.DataFrame()
        self.monte_carlo_samples = pd.DataFrame()
        self.battery_renewable_energy_kwh = 0.0

    def _build_economic_params(self, overrides=None):
        """Build lifecycle economics assumptions with user overrides."""
        defaults = {
            'currency': 'USD',
            'project_lifetime_years': 20,
            'nominal_discount_rate': 0.12,
            'general_inflation_rate': 0.03,
            'inflation_volatility': 0.02,
            'inflation_ar1': 0.45,
            'base_exchange_rate': 1.0,
            'exchange_rate_volatility': 0.08,
            'exchange_rate_ar1': 0.35,
            'fuel_price_per_liter': 1.50,
            'fuel_price_escalation_rate': 0.05,
            'fuel_price_volatility': 0.18,
            'energy_tariff_per_kwh': 0.30,
            'tariff_escalation_rate': 0.03,
            'pv_capex_per_kwp': 900.0,
            'wind_capex_per_kw': 1500.0,
            'hydro_capex_per_kw': 2500.0,
            'diesel_capex_per_kw': 550.0,
            'battery_capex_per_kwh': 350.0,
            'battery_power_capex_per_kw': 150.0,
            'pv_fixed_om_per_kw_year': 18.0,
            'wind_fixed_om_per_kw_year': 45.0,
            'hydro_fixed_om_per_kw_year': 35.0,
            'diesel_fixed_om_per_kw_year': 20.0,
            'diesel_maintenance_cost_per_hour': 1.50,
            'battery_fixed_om_per_kwh_year': 8.0,
            'battery_variable_om_per_kwh': 0.01,
            'diesel_variable_om_per_kwh': 0.03,
            'unserved_energy_cost_per_kwh': 2.00,
            'pv_capex_escalation_rate': 0.03,
            'wind_capex_escalation_rate': 0.03,
            'hydro_capex_escalation_rate': 0.03,
            'diesel_capex_escalation_rate': 0.03,
            'battery_capex_escalation_rate': 0.03,
            'om_escalation_rate': 0.03,
            'battery_replacement_cost_fraction': 0.80,
            'diesel_replacement_cost_fraction': 1.00,
            'pv_replacement_cost_fraction': 1.00,
            'wind_replacement_cost_fraction': 1.00,
            'hydro_replacement_cost_fraction': 0.90,
            'inverter_replacement_cost_fraction': 0.12,
            'include_salvage_value': True,
            'debt_fraction': 0.70,
            'debt_interest_rate': 0.10,
            'debt_tenor_years': 10,
            'monte_carlo_runs': 200,
        }
        if overrides:
            defaults.update(overrides)
        return defaults

    def _build_dsm_params(self, load_params=None):
        """Build demand-side management configuration."""
        defaults = {
            'enable_dsm': False,
            'deferrable_load_fraction': 0.0,
            'peak_reduction_fraction': 0.0,
            'peak_start_hour': 18,
            'peak_end_hour': 22,
            'shift_start_hour': 10,
            'shift_end_hour': 16,
        }
        if load_params:
            for key in defaults:
                if key in load_params:
                    defaults[key] = load_params[key]
        return defaults

    def _build_diesel_reliability_params(self, diesel_params=None):
        """Build diesel outage and repair assumptions."""
        defaults = {
            'enable_generator_reliability': False,
            'mtbf_hours': 500.0,
            'mttr_hours': 8.0,
            'planned_maintenance_interval_hours': 1000.0,
            'planned_maintenance_duration_hours': 6.0,
        }
        if diesel_params:
            for key in defaults:
                if key in diesel_params:
                    defaults[key] = diesel_params[key]
        return defaults

    def _annualize_value(self, value):
        """Convert simulated-period totals to annual equivalents."""
        return value / self.simulated_years

    def _discount_factor(self, year_index):
        """Nominal discount factor for year n cash flows."""
        rate = self.economic_params['nominal_discount_rate']
        return (1.0 + rate) ** year_index

    def _escalated_cost(self, base_cost, escalation_rate, year_index):
        """Escalate a base-year nominal cost into future year nominal terms."""
        return base_cost * ((1.0 + escalation_rate) ** max(year_index - 1, 0))

    def _estimate_battery_replacement_interval_years(self, annual_battery_throughput_kwh, annual_avg_battery_temp_c=25.0):
        """Estimate battery replacement interval from fade-to-EOL instead of simple cycle count."""
        nominal_capacity = max(self.battery.nominal_capacity_kwh, 1e-6)
        eol_fraction = getattr(self.battery, 'end_of_life_capacity_fraction', 0.80)
        fade_margin = max(1.0 - eol_fraction, 1e-6)
        temp_multiplier = self.battery._temperature_fade_multiplier(annual_avg_battery_temp_c)

        annual_calendar_fade = min(
            fade_margin,
            365.0 * self.battery.calendar_fade_rate * temp_multiplier
        )
        annual_cycle_fade = annual_battery_throughput_kwh * self.battery.cycle_fade_per_kwh * temp_multiplier

        throughput_limit_mwh = getattr(self.battery, 'lifetime_throughput_MWh', None)
        throughput_life = np.inf
        if throughput_limit_mwh is not None and annual_battery_throughput_kwh > 0:
            throughput_life = (throughput_limit_mwh * 1000.0) / annual_battery_throughput_kwh

        annual_total_fade = annual_calendar_fade + annual_cycle_fade
        fade_life = np.inf if annual_total_fade <= 0 else fade_margin / annual_total_fade
        calendar_life = max(self.battery.lifetime_years, 1.0)
        return max(1.0, min(calendar_life, fade_life, throughput_life))

    def _annual_debt_service(self, principal, interest_rate, tenor_years):
        """Level annual debt service for a simple annuity loan."""
        if principal <= 0 or tenor_years <= 0:
            return 0.0
        if interest_rate <= 0:
            return principal / tenor_years
        growth = (1.0 + interest_rate) ** tenor_years
        return principal * interest_rate * growth / max(growth - 1.0, 1e-9)

    def _generate_stochastic_financial_paths(self, project_life, rng):
        """Generate annual stochastic fuel, inflation, and FX paths."""
        econ = self.economic_params
        fuel_prices = np.zeros(project_life + 1)
        cpi_index = np.ones(project_life + 1)
        fx_index = np.ones(project_life + 1)

        fuel_prices[0] = econ['fuel_price_per_liter']
        inflation_mean = econ['general_inflation_rate']
        inflation_phi = econ['inflation_ar1']
        inflation_dev = 0.0

        fx_phi = econ['exchange_rate_ar1']
        log_fx_dev = 0.0

        for year in range(1, project_life + 1):
            fuel_drift = np.log1p(econ['fuel_price_escalation_rate'])
            fuel_shock = econ['fuel_price_volatility'] * rng.normal()
            fuel_prices[year] = fuel_prices[year - 1] * np.exp(fuel_drift - 0.5 * econ['fuel_price_volatility'] ** 2 + fuel_shock)

            inflation_dev = inflation_phi * inflation_dev + econ['inflation_volatility'] * rng.normal()
            inflation_rate = max(-0.90, inflation_mean + inflation_dev)
            cpi_index[year] = cpi_index[year - 1] * (1.0 + inflation_rate)

            log_fx_dev = fx_phi * log_fx_dev + econ['exchange_rate_volatility'] * rng.normal()
            fx_index[year] = np.exp(log_fx_dev)

        return {
            'fuel_price_per_liter': fuel_prices,
            'cpi_index': cpi_index,
            'fx_index': fx_index,
        }

    def _build_lifecycle_cashflow(self, annual_metrics, stochastic_paths=None):
        """Construct discounted lifecycle economics from annualized simulation outputs."""
        econ = self.economic_params
        project_life = int(max(1, econ['project_lifetime_years']))

        pv_capex = self.pv_gen.array_capacity_kwp * econ['pv_capex_per_kwp']
        wind_capex = self.wind_turbine.rated_power_kw * econ['wind_capex_per_kw']
        hydro_capex = self.hydro_turbine.rated_power_kw * econ['hydro_capex_per_kw']
        diesel_capex = self.diesel_gen.rated_kw * econ['diesel_capex_per_kw']
        battery_capex = (
            self.battery.energy_capacity_kwh * econ['battery_capex_per_kwh'] +
            self.battery.power_capacity_kw * econ['battery_power_capex_per_kw']
        )

        upfront_capex = pv_capex + wind_capex + hydro_capex + diesel_capex + battery_capex

        annual_fixed_om_base = (
            self.pv_gen.array_capacity_kwp * econ['pv_fixed_om_per_kw_year'] +
            self.wind_turbine.rated_power_kw * econ['wind_fixed_om_per_kw_year'] +
            self.hydro_turbine.rated_power_kw * econ['hydro_fixed_om_per_kw_year'] +
            self.diesel_gen.rated_kw * econ['diesel_fixed_om_per_kw_year'] +
            self.battery.energy_capacity_kwh * econ['battery_fixed_om_per_kwh_year']
        )

        annual_battery_throughput_kwh = annual_metrics['total_battery_throughput_kwh']
        annual_avg_battery_temp_c = annual_metrics.get('average_battery_temperature_c', 25.0)
        battery_replacement_interval = self._estimate_battery_replacement_interval_years(
            annual_battery_throughput_kwh,
            annual_avg_battery_temp_c
        )

        annual_runtime = annual_metrics['diesel_runtime_hours']
        diesel_runtime_limit = max(self.diesel_gen.end_of_life_hours, 1.0)
        debt_principal = upfront_capex * econ['debt_fraction']
        annual_debt_service = self._annual_debt_service(
            debt_principal,
            econ['debt_interest_rate'],
            int(max(econ['debt_tenor_years'], 0))
        )

        cashflows = []
        discounted_energy_served = 0.0
        discounted_cost_total = upfront_capex
        discounted_operating_cost = 0.0
        discounted_unserved_cost = 0.0
        discounted_salvage = 0.0

        cumulative_runtime = 0.0
        last_battery_install_year = 0
        last_diesel_install_year = 0
        last_pv_install_year = 0
        last_wind_install_year = 0
        last_hydro_install_year = 0

        for year in range(0, project_life + 1):
            if year == 0:
                cashflows.append({
                    'year': year,
                    'energy_served_kwh': 0.0,
                    'fixed_om_cost': 0.0,
                    'fuel_cost': 0.0,
                    'diesel_variable_om_cost': 0.0,
                    'diesel_maintenance_cost': 0.0,
                    'battery_variable_om_cost': 0.0,
                    'unserved_energy_cost': 0.0,
                    'replacement_cost': 0.0,
                    'salvage_value': 0.0,
                    'revenue': 0.0,
                    'cfads': 0.0,
                    'debt_service': 0.0,
                    'dscr': np.nan,
                    'capital_cost': upfront_capex,
                    'total_cost': upfront_capex,
                    'discounted_total_cost': upfront_capex,
                    'discounted_energy_served_kwh': 0.0,
                })
                continue

            if stochastic_paths is None:
                fuel_price = econ['fuel_price_per_liter'] * ((1.0 + econ['fuel_price_escalation_rate']) ** max(year - 1, 0))
                om_multiplier = (1.0 + econ['om_escalation_rate']) ** max(year - 1, 0)
                tariff = econ['energy_tariff_per_kwh'] * ((1.0 + econ['tariff_escalation_rate']) ** max(year - 1, 0))
                fx_multiplier = (1.0 + econ['general_inflation_rate']) ** max(year - 1, 0)
            else:
                fuel_price = float(stochastic_paths['fuel_price_per_liter'][year])
                om_multiplier = float(stochastic_paths['cpi_index'][year])
                tariff = econ['energy_tariff_per_kwh'] * float(stochastic_paths['cpi_index'][year])
                fx_multiplier = float(stochastic_paths['fx_index'][year])

            fuel_cost = annual_metrics['total_fuel_liters'] * fuel_price
            fixed_om_cost = annual_fixed_om_base * om_multiplier
            diesel_variable_om_cost = annual_metrics['total_diesel_generation_kwh'] * econ['diesel_variable_om_per_kwh'] * om_multiplier
            diesel_maintenance_cost = annual_runtime * econ['diesel_maintenance_cost_per_hour'] * om_multiplier
            battery_variable_om_cost = annual_metrics['total_battery_throughput_kwh'] * econ['battery_variable_om_per_kwh'] * om_multiplier
            unserved_energy_cost = annual_metrics['total_load_shedding_kwh'] * econ['unserved_energy_cost_per_kwh'] * om_multiplier
            revenue = annual_metrics['total_load_served_kwh'] * tariff

            replacement_cost = 0.0
            cumulative_runtime += annual_runtime

            if annual_runtime > 0 and cumulative_runtime >= diesel_runtime_limit:
                replacement_cost += diesel_capex * econ['diesel_replacement_cost_fraction'] * fx_multiplier
                cumulative_runtime = max(0.0, cumulative_runtime - diesel_runtime_limit)
                last_diesel_install_year = year

            if year - last_battery_install_year >= battery_replacement_interval:
                replacement_cost += battery_capex * econ['battery_replacement_cost_fraction'] * fx_multiplier
                last_battery_install_year = year

            if year - last_pv_install_year >= self.pv_gen.lifetime_years:
                replacement_cost += pv_capex * econ['pv_replacement_cost_fraction'] * fx_multiplier
                last_pv_install_year = year

            if year - last_wind_install_year >= self.wind_turbine.lifetime_years:
                replacement_cost += wind_capex * econ['wind_replacement_cost_fraction'] * fx_multiplier
                last_wind_install_year = year

            if year - last_hydro_install_year >= self.hydro_turbine.lifetime_years:
                replacement_cost += hydro_capex * econ['hydro_replacement_cost_fraction'] * fx_multiplier
                last_hydro_install_year = year

            if self.pv_gen.inverter_lifetime > 0 and year % self.pv_gen.inverter_lifetime == 0:
                replacement_cost += pv_capex * econ['inverter_replacement_cost_fraction'] * fx_multiplier

            salvage_value = 0.0
            if econ['include_salvage_value'] and year == project_life:
                battery_remaining_fraction = max(
                    0.0,
                    1.0 - ((project_life - last_battery_install_year) / battery_replacement_interval)
                )
                diesel_remaining_fraction = 0.0
                if annual_runtime > 0:
                    diesel_remaining_fraction = max(0.0, 1.0 - (cumulative_runtime / diesel_runtime_limit))
                pv_remaining_fraction = max(
                    0.0,
                    1.0 - ((project_life - last_pv_install_year) / max(self.pv_gen.lifetime_years, 1.0))
                )
                wind_remaining_fraction = max(
                    0.0,
                    1.0 - ((project_life - last_wind_install_year) / max(self.wind_turbine.lifetime_years, 1.0))
                )
                hydro_remaining_fraction = max(
                    0.0,
                    1.0 - ((project_life - last_hydro_install_year) / max(self.hydro_turbine.lifetime_years, 1.0))
                )

                salvage_value = (
                    battery_capex * fx_multiplier * battery_remaining_fraction +
                    diesel_capex * fx_multiplier * diesel_remaining_fraction +
                    pv_capex * fx_multiplier * pv_remaining_fraction +
                    wind_capex * fx_multiplier * wind_remaining_fraction +
                    hydro_capex * fx_multiplier * hydro_remaining_fraction
                )

            annual_total_cost = (
                fixed_om_cost + fuel_cost + diesel_variable_om_cost +
                diesel_maintenance_cost + battery_variable_om_cost +
                unserved_energy_cost + replacement_cost -
                salvage_value
            )
            cfads = revenue - annual_total_cost
            debt_service = annual_debt_service if year <= econ['debt_tenor_years'] else 0.0
            dscr = np.nan if debt_service <= 0 else cfads / debt_service

            discount_factor = self._discount_factor(year)
            discounted_total = annual_total_cost / discount_factor
            discounted_energy = annual_metrics['total_load_served_kwh'] / discount_factor

            discounted_cost_total += discounted_total
            discounted_operating_cost += (
                fixed_om_cost + fuel_cost + diesel_variable_om_cost +
                diesel_maintenance_cost + battery_variable_om_cost
            ) / discount_factor
            discounted_unserved_cost += unserved_energy_cost / discount_factor
            discounted_salvage += salvage_value / discount_factor
            discounted_energy_served += discounted_energy

            cashflows.append({
                'year': year,
                'energy_served_kwh': annual_metrics['total_load_served_kwh'],
                'fixed_om_cost': fixed_om_cost,
                'fuel_cost': fuel_cost,
                'diesel_variable_om_cost': diesel_variable_om_cost,
                'diesel_maintenance_cost': diesel_maintenance_cost,
                'battery_variable_om_cost': battery_variable_om_cost,
                'unserved_energy_cost': unserved_energy_cost,
                'replacement_cost': replacement_cost,
                'salvage_value': salvage_value,
                'revenue': revenue,
                'cfads': cfads,
                'debt_service': debt_service,
                'dscr': dscr,
                'capital_cost': 0.0,
                'total_cost': annual_total_cost,
                'discounted_total_cost': discounted_total,
                'discounted_energy_served_kwh': discounted_energy,
            })

        cashflow_df = pd.DataFrame(cashflows)
        lcoe = discounted_cost_total / max(discounted_energy_served, 1e-6)
        dscr_series = cashflow_df.loc[cashflow_df['debt_service'] > 0, 'dscr']
        return cashflow_df, {
            'upfront_capex': upfront_capex,
            'discounted_lifecycle_cost': discounted_cost_total,
            'discounted_operating_cost': discounted_operating_cost,
            'discounted_unserved_energy_cost': discounted_unserved_cost,
            'discounted_salvage_value': discounted_salvage,
            'discounted_energy_served_kwh': discounted_energy_served,
            'lcoe': lcoe,
            'project_lifetime_years': project_life,
            'annual_fixed_om_base': annual_fixed_om_base,
            'battery_replacement_interval_years': battery_replacement_interval,
            'debt_principal': debt_principal,
            'annual_debt_service': annual_debt_service,
            'average_dscr': float(dscr_series.mean()) if not dscr_series.empty else np.nan,
            'minimum_dscr': float(dscr_series.min()) if not dscr_series.empty else np.nan,
        }

    def _run_monte_carlo_financial_risk(self, annual_metrics):
        """Run Monte Carlo on annual economics using stochastic fuel, CPI, and FX paths."""
        project_life = int(max(1, self.economic_params['project_lifetime_years']))
        n_runs = int(max(0, self.economic_params.get('monte_carlo_runs', 0)))
        if n_runs <= 0:
            return pd.DataFrame(), pd.DataFrame()

        rng = np.random.default_rng(self.random_seed)
        samples = []
        for run in range(n_runs):
            stochastic_paths = self._generate_stochastic_financial_paths(project_life, rng)
            _, lifecycle_metrics = self._build_lifecycle_cashflow(
                annual_metrics,
                stochastic_paths=stochastic_paths
            )
            samples.append({
                'run': run + 1,
                'discounted_lifecycle_cost': lifecycle_metrics['discounted_lifecycle_cost'],
                'lcoe': lifecycle_metrics['lcoe'],
                'minimum_dscr': lifecycle_metrics['minimum_dscr'],
                'average_dscr': lifecycle_metrics['average_dscr'],
            })

        samples_df = pd.DataFrame(samples)
        summary_df = pd.DataFrame([
            {'metric': 'discounted_lifecycle_cost', 'mean': samples_df['discounted_lifecycle_cost'].mean(), 'p10': samples_df['discounted_lifecycle_cost'].quantile(0.10), 'p50': samples_df['discounted_lifecycle_cost'].quantile(0.50), 'p90': samples_df['discounted_lifecycle_cost'].quantile(0.90), 'std': samples_df['discounted_lifecycle_cost'].std(ddof=0)},
            {'metric': 'lcoe', 'mean': samples_df['lcoe'].mean(), 'p10': samples_df['lcoe'].quantile(0.10), 'p50': samples_df['lcoe'].quantile(0.50), 'p90': samples_df['lcoe'].quantile(0.90), 'std': samples_df['lcoe'].std(ddof=0)},
            {'metric': 'minimum_dscr', 'mean': samples_df['minimum_dscr'].mean(), 'p10': samples_df['minimum_dscr'].quantile(0.10), 'p50': samples_df['minimum_dscr'].quantile(0.50), 'p90': samples_df['minimum_dscr'].quantile(0.90), 'std': samples_df['minimum_dscr'].std(ddof=0)},
        ])
        return samples_df, summary_df
        
    def _create_time_index(self):
        """Create time index for simulation with specified timestep resolution."""
        print("Creating time index...")
        time_index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f"{self.timestep_minutes}min",
            inclusive='left'
        )
        print(f"  + Time index has {len(time_index)} timesteps")
        return time_index

    def _find_column(self, df, candidates):
        """Find the first matching column by normalized name."""
        normalized = {col.lower().strip(): col for col in df.columns}
        for candidate in candidates:
            key = candidate.lower().strip()
            if key in normalized:
                return normalized[key]
        return None

    def _extract_location_from_resource_profile(self, resource_df):
        """Update synthetic-resource location from resource profile metadata when available."""
        lat_col = self._find_column(resource_df, ['latitude', 'lat'])
        lon_col = self._find_column(resource_df, ['longitude', 'lon', 'lng'])
        if lat_col is not None:
            lat_values = resource_df[lat_col].dropna()
            if not lat_values.empty:
                self.location_lat = float(lat_values.iloc[0])
        if lon_col is not None:
            lon_values = resource_df[lon_col].dropna()
            if not lon_values.empty:
                self.location_lon = float(lon_values.iloc[0])

    def _hour_mask(self, index, start_hour, end_hour):
        """Return a boolean mask for hours in a possibly wrap-around interval."""
        hours = index.hour + index.minute / 60.0
        if start_hour <= end_hour:
            return (hours >= start_hour) & (hours < end_hour)
        return (hours >= start_hour) | (hours < end_hour)

    def _apply_dsm(self, load_series):
        """Apply simple DSM actions: deferrable-load shifting and peak shaving."""
        adjusted = load_series.astype(float).copy()
        dsm_shifted_kw = pd.Series(0.0, index=load_series.index, name='dsm_shifted_load_kw')
        dsm_peak_reduced_kw = pd.Series(0.0, index=load_series.index, name='dsm_peak_reduction_kw')

        if not self.dsm_params['enable_dsm']:
            self.dsm_summary = {
                'total_shifted_load_kwh': 0.0,
                'total_peak_reduced_kwh': 0.0,
                'peak_load_before_dsm_kw': float(load_series.max()),
                'peak_load_after_dsm_kw': float(load_series.max()),
            }
            return pd.DataFrame({
                'baseline_load_kw': load_series,
                'load_kw': adjusted,
                'dsm_shifted_load_kw': dsm_shifted_kw,
                'dsm_peak_reduction_kw': dsm_peak_reduced_kw,
            })

        peak_mask = self._hour_mask(
            load_series.index,
            self.dsm_params['peak_start_hour'],
            self.dsm_params['peak_end_hour'],
        )
        shift_mask = self._hour_mask(
            load_series.index,
            self.dsm_params['shift_start_hour'],
            self.dsm_params['shift_end_hour'],
        )

        total_shifted_kwh = 0.0
        for day in pd.Index(load_series.index.normalize().unique()):
            day_mask = load_series.index.normalize() == day
            day_peak_idx = load_series.index[day_mask & peak_mask]
            day_shift_idx = load_series.index[day_mask & shift_mask]

            if len(day_peak_idx) == 0 or len(day_shift_idx) == 0:
                continue

            shift_fraction = float(np.clip(self.dsm_params['deferrable_load_fraction'], 0.0, 0.95))
            shift_down_kw = adjusted.loc[day_peak_idx] * shift_fraction
            shift_energy_kwh = float(shift_down_kw.sum() * self.timestep_hours)
            if shift_energy_kwh <= 0:
                continue

            adjusted.loc[day_peak_idx] = adjusted.loc[day_peak_idx] - shift_down_kw
            dsm_shifted_kw.loc[day_peak_idx] = dsm_shifted_kw.loc[day_peak_idx] - shift_down_kw

            shift_up_kw = shift_energy_kwh / (len(day_shift_idx) * self.timestep_hours)
            adjusted.loc[day_shift_idx] = adjusted.loc[day_shift_idx] + shift_up_kw
            dsm_shifted_kw.loc[day_shift_idx] = dsm_shifted_kw.loc[day_shift_idx] + shift_up_kw
            total_shifted_kwh += shift_energy_kwh

        peak_reduction_fraction = float(np.clip(self.dsm_params['peak_reduction_fraction'], 0.0, 0.95))
        if peak_reduction_fraction > 0:
            reduction_kw = adjusted.loc[peak_mask] * peak_reduction_fraction
            adjusted.loc[peak_mask] = adjusted.loc[peak_mask] - reduction_kw
            dsm_peak_reduced_kw.loc[peak_mask] = reduction_kw

        adjusted = adjusted.clip(lower=0.0)
        total_peak_reduced_kwh = float(dsm_peak_reduced_kw.sum() * self.timestep_hours)
        self.dsm_summary = {
            'total_shifted_load_kwh': total_shifted_kwh,
            'total_peak_reduced_kwh': total_peak_reduced_kwh,
            'peak_load_before_dsm_kw': float(load_series.max()),
            'peak_load_after_dsm_kw': float(adjusted.max()),
        }

        return pd.DataFrame({
            'baseline_load_kw': load_series,
            'load_kw': adjusted,
            'dsm_shifted_load_kw': dsm_shifted_kw,
            'dsm_peak_reduction_kw': dsm_peak_reduced_kw,
        })

    def _get_diesel_availability(self):
        """Return diesel availability status and advance outage timers."""
        if self.diesel_gen.rated_kw <= 0:
            return False, "not_installed"

        reliability = self.diesel_reliability
        if not reliability['enable_generator_reliability']:
            return True, "available"

        if self.diesel_outage_remaining_steps > 0:
            self.diesel_outage_remaining_steps -= 1
            if self.diesel_outage_remaining_steps <= 0:
                self.diesel_outage_reason = "available"
                return True, "available"
            return False, self.diesel_outage_reason

        if (
            reliability['planned_maintenance_interval_hours'] > 0 and
            self.diesel_gen.runtime_hours >= self.next_planned_maintenance_runtime_hours
        ):
            duration_steps = max(
                1,
                int(np.ceil(reliability['planned_maintenance_duration_hours'] / max(self.timestep_hours, 1e-9)))
            )
            self.diesel_outage_remaining_steps = duration_steps - 1
            self.diesel_outage_reason = "planned_maintenance"
            self.diesel_planned_outage_events += 1
            self.next_planned_maintenance_runtime_hours += reliability['planned_maintenance_interval_hours']
            return False, self.diesel_outage_reason

        mtbf_hours = reliability['mtbf_hours']
        if mtbf_hours > 0:
            outage_probability = min(1.0, self.timestep_hours / mtbf_hours)
            if np.random.random() < outage_probability:
                duration_steps = max(
                    1,
                    int(np.ceil(reliability['mttr_hours'] / max(self.timestep_hours, 1e-9)))
                )
                self.diesel_outage_remaining_steps = duration_steps - 1
                self.diesel_outage_reason = "forced_outage"
                self.diesel_forced_outage_events += 1
                return False, self.diesel_outage_reason

        return True, "available"

    def _load_resource_profile(self):
        """Load timestamped meteorological/hydro resource data if provided."""
        if not self.resource_profile_file:
            return None

        resource_path = Path(self.resource_profile_file)
        resource_df = pd.read_csv(resource_path)

        timestamp_col = self._find_column(resource_df, ['timestamp', 'datetime', 'date_time', 'time'])
        if timestamp_col is None:
            raise ValueError(f"No timestamp column found in {resource_path}")

        resource_df[timestamp_col] = pd.to_datetime(resource_df[timestamp_col])
        self._extract_location_from_resource_profile(resource_df)
        resource_df = resource_df.set_index(timestamp_col).sort_index()
        resource_df = resource_df[~resource_df.index.duplicated(keep='first')]

        resource_df = resource_df.reindex(self.time_index)
        resource_df = resource_df.interpolate(method='time').bfill().ffill()
        return resource_df
    
    def _generate_solar_data(self, resource_df=None):
        """Generate solar irradiance data using the internal solar resource pipeline."""
        print("Generating solar irradiance data...")

        if resource_df is not None:
            ghi_col = self._find_column(resource_df, [
                'ghi_w_m2', 'ghi', 'solar_irradiance_wm2', 'global_horizontal_irradiance_w_m2'
            ])
            if ghi_col is not None:
                irradiance = resource_df[ghi_col].astype(float).to_numpy()
                print(f"  + Solar irradiance loaded from profile: {ghi_col}")
                print(f"  + Solar irradiance mean: {np.mean(irradiance):.1f} W/m2")
                print(f"  + Solar irradiance max: {np.max(irradiance):.1f} W/m2")
                return irradiance

        latitude_scale = min(abs(self.location_lat) / 45.0, 1.0)
        seasonal_bias = -0.02 * latitude_scale if self.location_lat >= 0 else 0.02 * latitude_scale
        monthly_kt = np.clip(np.array([
            0.545, 0.562, 0.553, 0.560, 0.559, 0.591,
            0.581, 0.559, 0.566, 0.545, 0.536, 0.535
        ]) + seasonal_bias, 0.35, 0.75)

        sim = SolarResourceSimulator(lat=self.location_lat, lon=self.location_lon)
        hourly_irradiance = np.array(sim.run_full_year(monthly_kt.tolist()))

        # Resample to match the simulation timestep if not hourly
        if self.timestep_minutes != 60:
            original_idx = np.arange(len(hourly_irradiance))
            target_idx = np.linspace(0, len(hourly_irradiance) - 1, len(self.time_index))
            irradiance = np.interp(target_idx, original_idx, hourly_irradiance)
        else:
            irradiance = hourly_irradiance

        print(f"  + Solar irradiance mean: {np.mean(irradiance):.1f} W/m2")
        print(f"  + Solar irradiance max: {np.max(irradiance):.1f} W/m2")
        return irradiance
    
    def _generate_wind_data(self, resource_df=None, method='synthetic'):
        """
        Generate wind speed data over simulation horizon.
        
        Args:
            method (str): 'synthetic' (Weibull-based) or 'random'
        
        Returns:
            numpy array of wind speeds (m/s)
        """
        print("Generating wind speed data...")

        if resource_df is not None:
            wind_col = self._find_column(resource_df, [
                'wind_speed_ms', 'wind_speed_m_s', 'wind_speed', 'wind_ms'
            ])
            if wind_col is not None:
                wind_speed = resource_df[wind_col].astype(float).to_numpy()
                print(f"  + Wind speed loaded from profile: {wind_col}")
                print(f"  + Wind speed mean: {np.mean(wind_speed):.1f} m/s")
                print(f"  + Wind speed max: {np.max(wind_speed):.1f} m/s")
                return wind_speed
        
        if method == 'synthetic':
            # Weibull distribution with daily and seasonal variation
            day_of_year = np.array([d.dayofyear for d in self.time_index])
            
            hemisphere_shift = 0 if self.location_lat >= 0 else 182
            seasonal_mean = 7 + 3 * np.sin((day_of_year - 80 - hemisphere_shift) * 2 * np.pi / 365)
            
            # Daily variation (wind often higher during day)
            hour_of_day = np.array([d.hour for d in self.time_index])
            daily_variation = 1.0 + 0.3 * np.sin((hour_of_day - 6) * np.pi / 12)
            
            # Generate from Weibull distribution (shape=2, realistic for wind)
            wind_speed = np.random.weibull(2.0, len(self.time_index))
            wind_speed = wind_speed * seasonal_mean * daily_variation / 2.0
            
        else:  # random method
            # Simple AR(1) process
            wind_speed = np.zeros(len(self.time_index))
            wind_speed[0] = np.random.exponential(7)
            
            for i in range(1, len(self.time_index)):
                wind_speed[i] = 0.8 * wind_speed[i-1] + np.random.normal(3, 2)
                wind_speed[i] = np.maximum(0, np.minimum(20, wind_speed[i]))
        
        print(f"  + Wind speed mean: {np.mean(wind_speed):.1f} m/s")
        print(f"  + Wind speed max: {np.max(wind_speed):.1f} m/s")
        return wind_speed

    def _dumortier_hourly_temperature(self, daily_tmin, daily_tmax):
        """Construct hourly temperatures from daily Tmin/Tmax using an irradiance-ratio shape."""
        temperature = np.zeros(len(self.time_index))

        for day_idx, day in enumerate(pd.Index(self.time_index.normalize().unique())):
            day_mask = self.time_index.normalize() == day
            day_positions = np.flatnonzero(day_mask)
            if len(day_positions) == 0:
                continue

            tmin = daily_tmin[min(day_idx, len(daily_tmin) - 1)]
            tmax = daily_tmax[min(day_idx, len(daily_tmax) - 1)]

            g0h = []
            for pos in day_positions:
                timestamp = self.time_index[pos]
                hour = timestamp.hour + timestamp.minute / 60.0
                extraterrestrial, _, _ = extraterrestrial_hour(timestamp.dayofyear, hour)
                g0h.append(max(extraterrestrial, 0.0))

            g0h = np.array(g0h, dtype=float)
            daylight_total = g0h[g0h > 0].sum()
            if daylight_total > 0:
                kx = np.where(g0h > 0, g0h / daylight_total, 0.0)
                daylight_mask = g0h > 0
                daylight_scale = kx[daylight_mask] / max(kx[daylight_mask].max(), 1e-9)
                day_values = np.full(len(day_positions), tmin, dtype=float)
                day_values[daylight_mask] = tmin + (tmax - tmin) * daylight_scale
            else:
                day_values = np.full(len(day_positions), tmin, dtype=float)

            night_mask = g0h <= 0
            if np.any(night_mask):
                night_indices = np.flatnonzero(night_mask)
                for idx in night_indices:
                    frac = idx / max(len(day_positions) - 1, 1)
                    day_values[idx] = tmin + 0.15 * (tmax - tmin) * (1.0 - frac)

            temperature[day_positions] = day_values

        return temperature
    
    def _generate_temperature_data(self, resource_df=None):
        """
        Generate ambient temperature data over simulation horizon.
        
        Returns:
            numpy array of temperatures (°C)
        """
        print("Generating temperature data...")

        if resource_df is not None:
            temp_col = self._find_column(resource_df, [
                'temperature_c', 'ambient_temp_c', 'temp_c', 'temperature'
            ])
            if temp_col is not None:
                temperature = resource_df[temp_col].astype(float).to_numpy()
                print(f"  + Temperature loaded from profile: {temp_col}")
                print(f"  + Temperature mean: {np.mean(temperature):.1f} C")
                print(f"  + Temperature range: [{np.min(temperature):.1f}, {np.max(temperature):.1f}] C")
                return temperature

            tmin_col = self._find_column(resource_df, ['tmin_c', 'temperature_min_c', 'daily_tmin_c'])
            tmax_col = self._find_column(resource_df, ['tmax_c', 'temperature_max_c', 'daily_tmax_c'])
            if tmin_col is not None and tmax_col is not None:
                daily_temp = resource_df[[tmin_col, tmax_col]].copy()
                daily_temp['day'] = self.time_index.normalize()
                daily_grouped = daily_temp.groupby('day').mean(numeric_only=True)
                temperature = self._dumortier_hourly_temperature(
                    daily_grouped[tmin_col].to_numpy(),
                    daily_grouped[tmax_col].to_numpy()
                )
                print(f"  + Temperature reconstructed from daily Tmin/Tmax using irradiance-ratio profile")
                print(f"  + Temperature mean: {np.mean(temperature):.1f} C")
                print(f"  + Temperature range: [{np.min(temperature):.1f}, {np.max(temperature):.1f}] C")
                return temperature

        days = pd.Index(self.time_index.normalize().unique())
        day_of_year = np.array([day.dayofyear for day in days])

        # Synthetic daily extrema before hourly reconstruction.
        hemisphere_shift = 0 if self.location_lat >= 0 else 182
        latitude_temp_offset = -0.12 * abs(self.location_lat)
        daily_tmin = (
            12 + latitude_temp_offset +
            6 * np.sin((day_of_year - 110 - hemisphere_shift) * 2 * np.pi / 365) +
            np.random.normal(0, 1.0, len(days))
        )
        daily_tmax = (
            daily_tmin + 8 +
            3 * np.sin((day_of_year - 80 - hemisphere_shift) * 2 * np.pi / 365) +
            np.random.normal(0, 1.0, len(days))
        )
        daily_tmax = np.maximum(daily_tmax, daily_tmin + 1.0)

        temperature = self._dumortier_hourly_temperature(daily_tmin, daily_tmax)
        temperature += np.random.normal(0, 0.5, len(self.time_index))
        
        print(f"  + Temperature mean: {np.mean(temperature):.1f} C")
        print(f"  + Temperature range: [{np.min(temperature):.1f}, {np.max(temperature):.1f}] C")
        return temperature
    
    def _generate_load_data(self, resource_df=None):
        """Generate load time series for simulation horizon.

        If a load-profile CSV is provided, that profile is resampled to the
        selected timestep and repeated across the full simulation year.
        Otherwise a synthetic load profile is generated from the internal demand model.
        """
        print("Generating load profile...")

        if self.load_profile_file:
            load_series = self._load_provided_profile()
        elif resource_df is not None:
            load_col = self._find_column(resource_df, ['load_kw', 'load', 'demand_kw', 'load_demand_kw'])
            if load_col is not None:
                load_series = pd.Series(resource_df[load_col].astype(float).values, index=self.time_index, name='Load_kW')
            else:
                load_series = self.load_model.generate_load(
                    year=self.start_date.year,
                    num_days=self.num_days
                )
        else:
            load_series = self.load_model.generate_load(
                year=self.start_date.year,
                num_days=self.num_days
            )

        # Align series length/indices to simulation time index
        if len(load_series) != len(self.time_index):
            # Resample/interpolate if needed
            load_series = load_series.reindex(self.time_index, method='nearest', fill_value=np.nan)
            load_series = load_series.interpolate(method='time').bfill().ffill()

        load_df = self._apply_dsm(load_series)
        load_values = load_df['load_kw'].values

        print(f"  + Load mean: {np.mean(load_values):.1f} kW")
        print(f"  + Load max: {np.max(load_values):.1f} kW")
        print(f"  + Load min: {np.min(load_values):.1f} kW")
        if self.dsm_params['enable_dsm']:
            print(f"  + DSM shifted energy: {self.dsm_summary['total_shifted_load_kwh']:.1f} kWh")
            print(f"  + DSM peak reduction: {self.dsm_summary['total_peak_reduced_kwh']:.1f} kWh")
        return load_df

    def _generate_hydro_data(self, resource_df=None):
        """Generate or load hydro flow/head data for the simulation horizon."""
        print("Generating hydropower resource data...")

        if resource_df is not None:
            flow_col = self._find_column(resource_df, ['hydro_flow_m3s', 'flow_m3s', 'river_flow_m3s'])
            head_col = self._find_column(resource_df, ['hydro_head_m', 'head_m', 'net_head_m'])
            if flow_col is not None:
                flow_m3s = resource_df[flow_col].astype(float).to_numpy()
                if head_col is not None:
                    head_m = resource_df[head_col].astype(float).to_numpy()
                else:
                    head_m = np.full(len(self.time_index), self.hydro_turbine.net_head_m)
                print(f"  + Hydro flow loaded from profile: {flow_col}")
                print(f"  + Hydro flow mean: {np.mean(flow_m3s):.2f} m3/s")
                return flow_m3s, head_m

        flow_m3s = np.full(len(self.time_index), self.hydro_turbine.design_flow_m3s)
        seasonal_factor = 1.0 + 0.35 * np.sin((np.array([d.dayofyear for d in self.time_index]) - 120) * 2 * np.pi / 365)
        flow_m3s = np.maximum(0.0, flow_m3s * seasonal_factor)
        head_m = np.full(len(self.time_index), self.hydro_turbine.net_head_m)
        print(f"  + Hydro flow mean: {np.mean(flow_m3s):.2f} m3/s")
        return flow_m3s, head_m

    def _load_provided_profile(self):
        """Load a provided profile and tile it across the simulation horizon.

        Assumption:
        - If the CSV only contains a representative day, that day is repeated
          across the simulation horizon.
        """
        profile_path = Path(self.load_profile_file)
        profile_df = pd.read_csv(profile_path)

        load_col = next(
            (col for col in profile_df.columns if 'load' in col.lower()),
            None
        )
        if load_col is None:
            raise ValueError(f"No load column found in {profile_path}")

        if 'Time' in profile_df.columns:
            times = pd.to_timedelta(profile_df['Time'].astype(str) + ':00')
            profile_index = pd.Timestamp('2000-01-01') + times
        else:
            base_freq = pd.to_timedelta(self.timestep_minutes, unit='min')
            profile_index = pd.date_range(
                start='2000-01-01',
                periods=len(profile_df),
                freq=base_freq
            )

        daily_profile = pd.Series(
            profile_df[load_col].astype(float).values,
            index=pd.DatetimeIndex(profile_index),
            name='Load_kW'
        ).sort_index()

        simulation_minutes = (
            self.time_index.hour * 60 + self.time_index.minute
        ).to_numpy()
        source_minutes = (
            (daily_profile.index - daily_profile.index.normalize())
            .total_seconds() / 60.0
        ).to_numpy()
        source_load = daily_profile.values.astype(float)

        # Extend the representative day so interpolation works near midnight.
        source_minutes = np.concatenate((
            [source_minutes[-1] - 1440.0],
            source_minutes,
            [source_minutes[0] + 1440.0]
        ))
        source_load = np.concatenate((
            [source_load[-1]],
            source_load,
            [source_load[0]]
        ))

        tiled_load = np.interp(
            simulation_minutes,
            source_minutes,
            source_load
        )

        return pd.Series(tiled_load, index=self.time_index, name='Load_kW')
    
    def run_simulation(self, save_results=True, verbose=True):
        """
        Run the complete microgrid simulation over the time horizon.
        
        Args:
            save_results (bool): Whether to save results to CSV
            verbose (bool): Whether to print detailed timestep information
        
        Returns:
            pandas DataFrame with complete simulation results
        """
        print("\n" + "="*70)
        print("RUNNING MICROGRID SIMULATION")
        print("="*70 + "\n")
        
        # Generate resource and load data
        resource_df = self._load_resource_profile()
        solar_irradiance = self._generate_solar_data(resource_df=resource_df)
        wind_speed = self._generate_wind_data(resource_df=resource_df)
        temperature = self._generate_temperature_data(resource_df=resource_df)
        hydro_flow_m3s, hydro_head_m = self._generate_hydro_data(resource_df=resource_df)
        load_demand = self._generate_load_data(resource_df=resource_df)
        print()
        
        self.results = []
        self.battery_renewable_energy_kwh = 0.0
        self.diesel_outage_remaining_steps = 0
        self.diesel_outage_reason = "available"
        self.next_planned_maintenance_runtime_hours = self.diesel_reliability['planned_maintenance_interval_hours']
        self.diesel_forced_outage_events = 0
        self.diesel_planned_outage_events = 0

        # Run timestep-by-timestep simulation
        print("Executing timestep dispatch...")
        
        for step, timestamp in enumerate(self.time_index):
            if verbose and step % max(1, self.steps_per_day) == 0:
                print(f"  Step {step+1}/{self.total_steps} ({timestamp.date()})")
            
            # Step 1: Get resource data
            pv_state = pv_power_from_ghi(
                timestamp=timestamp,
                ghi_w_m2=solar_irradiance[step],
                ambient_temp_c=temperature[step],
                system_size_w=self.pv_gen.array_capacity_kwp * 1000.0,
                isc_temp_coeff_rel=self.pv_model_params['isc_temp_coeff_rel']
            )
            solar_kw = pv_state['pv_power_w'] / 1000.0
            
            wind_kw = self.wind_turbine.power_output(
                wind_speed=wind_speed[step],
                temp_c=temperature[step]
            )
            hydro_kw = self.hydro_turbine.power_output(
                flow_m3s=hydro_flow_m3s[step],
                head_m=hydro_head_m[step]
            )
            
            # Step 2: Get load
            load_row = load_demand.iloc[step]
            load_kw = float(load_row['load_kw'])
            baseline_load_kw = float(load_row['baseline_load_kw'])
            dsm_shifted_load_kw = float(load_row['dsm_shifted_load_kw'])
            dsm_peak_reduction_kw = float(load_row['dsm_peak_reduction_kw'])
            battery_energy_before_kwh = self.battery.get_energy_kwh()
            renewable_share_before = 0.0
            if battery_energy_before_kwh > 1e-9:
                renewable_share_before = min(
                    1.0,
                    max(0.0, self.battery_renewable_energy_kwh / battery_energy_before_kwh)
                )

            diesel_available, diesel_outage_reason = self._get_diesel_availability()
            
            # Step 3: Run dispatch algorithm (Priority: Solar → Wind → Battery → Diesel)
            dispatch = load_following_algorithm(
                load_kw=load_kw,
                solar_power_kw=solar_kw,
                wind_power_kw=wind_kw,
                hydro_power_kw=hydro_kw,
                battery=self.battery,
                diesel_generator=self.diesel_gen,
                diesel_available=diesel_available,
                battery_variable_cost_per_kwh=self.economic_params['battery_variable_om_per_kwh'],
                diesel_fuel_price_per_liter=self.economic_params['fuel_price_per_liter'],
                diesel_variable_om_per_kwh=self.economic_params['diesel_variable_om_per_kwh'],
                diesel_maintenance_cost_per_hour=self.economic_params['diesel_maintenance_cost_per_hour'],
                ambient_temp_c=temperature[step],
                timestep_hr=self.timestep_hours,
                dispatch_strategy=self.dispatch_strategy
            )
            
            # Step 4: Update system states (battery aging, equipment degradation, etc.)
            self._update_system_states(step)

            # Step 5: Record energy balance error for debugging
            power_balance_kw = dispatch.get('error_kw', 0.0)

            # Step 6: Store results
            battery_discharge_kw = dispatch.get('battery_discharge_kwh', 0.0) / self.timestep_hours
            battery_charge_kw = dispatch.get('battery_charge_kwh', 0.0) / self.timestep_hours
            renewable_served_kwh = dispatch.get('renewable_served_kwh', 0.0)
            battery_discharge_kwh = dispatch.get('battery_discharge_kwh', 0.0)
            battery_charge_kwh = dispatch.get('battery_charge_kwh', 0.0)
            diesel_generation_kwh = dispatch.get('diesel', 0.0) * self.timestep_hours
            diesel_served_kwh = dispatch.get('diesel_served_kwh', 0.0)

            renewable_battery_served_kwh = battery_discharge_kwh * renewable_share_before
            stored_renewable_removed_kwh = (
                battery_discharge_kwh / max(self.battery.discharge_efficiency, 1e-9)
            ) * renewable_share_before
            self.battery_renewable_energy_kwh = max(
                0.0,
                self.battery_renewable_energy_kwh - stored_renewable_removed_kwh
            )

            renewable_surplus_kwh = max(0.0, solar_kw * self.timestep_hours + wind_kw * self.timestep_hours + hydro_kw * self.timestep_hours - renewable_served_kwh)
            diesel_surplus_kwh = max(0.0, diesel_generation_kwh - diesel_served_kwh)
            renewable_charge_input_kwh = min(battery_charge_kwh, renewable_surplus_kwh)
            diesel_charge_input_kwh = min(
                max(0.0, battery_charge_kwh - renewable_charge_input_kwh),
                diesel_surplus_kwh
            )
            stored_renewable_added_kwh = renewable_charge_input_kwh * self.battery.charge_efficiency
            self.battery_renewable_energy_kwh = min(
                self.battery.get_energy_kwh(),
                self.battery_renewable_energy_kwh + stored_renewable_added_kwh
            )
            total_renewable_served_kwh = renewable_served_kwh + renewable_battery_served_kwh

            timestep_result = {
                'timestamp': timestamp,
                'step': step,
                'hour_of_year': timestamp.hour + (timestamp.dayofyear - 1) * 24,

                # Demand
                'baseline_load_kw': baseline_load_kw,
                'load_kw': load_kw,
                'dsm_shifted_load_kw': dsm_shifted_load_kw,
                'dsm_peak_reduction_kw': dsm_peak_reduction_kw,
                'load_served_kw': dispatch.get('load_served_kw', 0.0),
                'load_shedding_kw': dispatch.get('load_shedding_kw', 0.0),
                'renewable_served_kw': dispatch.get('renewable_served_kwh', 0.0) / self.timestep_hours,
                'renewable_battery_served_kw': renewable_battery_served_kwh / self.timestep_hours,
                'total_renewable_served_kw': total_renewable_served_kwh / self.timestep_hours,
                'battery_served_kw': dispatch.get('battery_served_kwh', 0.0) / self.timestep_hours,
                'diesel_served_kw': dispatch.get('diesel_served_kwh', 0.0) / self.timestep_hours,

                # Resource availability
                'solar_irradiance_wm2': solar_irradiance[step],
                'tilted_irradiance_wm2': pv_state['tilted_irradiance_w_m2'],
                'cell_temperature_c': pv_state['cell_temp_c'],
                'pv_module_count': pv_state['module_count'],
                'wind_speed_ms': wind_speed[step],
                'temperature_c': temperature[step],
                'hydro_flow_m3s': hydro_flow_m3s[step],
                'hydro_head_m': hydro_head_m[step],

                # Generation
                'solar_generation_kw': solar_kw,
                'wind_generation_kw': wind_kw,
                'hydro_generation_kw': hydro_kw,
                'diesel_generation_kw': dispatch.get('diesel', 0.0),
                'battery_discharge_kw': battery_discharge_kw,
                'battery_charge_kw': battery_charge_kw,
                'battery_charge_from_renewables_kw': renewable_charge_input_kwh / self.timestep_hours,
                'battery_charge_from_diesel_kw': diesel_charge_input_kwh / self.timestep_hours,
                'curtailment_kw': dispatch.get('curtailment', 0.0),
                'total_generation_kw': (solar_kw + wind_kw + hydro_kw + dispatch.get('diesel', 0.0)),
                'total_supply_kw': (solar_kw + wind_kw + hydro_kw + dispatch.get('diesel', 0.0) + battery_discharge_kw),

                # Energy balance check
                'power_balance_kw': power_balance_kw,

                # Costs
                'operating_cost': dispatch.get('operating_cost', 0.0),
                'fuel_liters': dispatch.get('fuel_liters', 0.0),
                'fuel_cost': dispatch.get('fuel_liters', 0.0) * self.economic_params['fuel_price_per_liter'],

                # Equipment status
                'diesel_available': diesel_available,
                'diesel_outage_reason': diesel_outage_reason,
                'pv_operating_years': self.pv_gen.operating_years,
                'diesel_operating_hours': self.diesel_gen.runtime_hours,
                'battery_soc_before': dispatch.get('battery_soc_before', np.nan),
                'battery_soc_after': dispatch.get('battery_soc_after', np.nan),
                'battery_health': self.battery.current_capacity_kwh / self.battery.nominal_capacity_kwh,
                'battery_temperature_c': self.battery.current_temp_c,
                'battery_renewable_energy_kwh': self.battery_renewable_energy_kwh,
            }

            self.results.append(timestep_result)
        
        print(f"\n  + Simulation complete: {len(self.results)} timesteps")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Calculate and display performance metrics
        self._calculate_performance_metrics(results_df)
        
        # Save results
        if save_results:
            self._save_results(results_df)
        
        return results_df
    
    def _update_system_states(self, step):
        """Update degradation and state of system components."""
        # Update PV degradation (yearly)
        if (step + 1) % (self.steps_per_day * 365) == 0:
            self.pv_gen.step_year()

        # Update diesel runtime hours (tracked in fuel consumption)
        # Diesel runtime is incremented in DieselGenerator.fuel_consumption

        # Battery calendar aging (apply once per day)
        if (step + 1) % self.steps_per_day == 0:
            daily_throughput_kwh = max(
                0.0,
                (self.battery.total_throughput_mwh - self._last_battery_throughput_mwh) * 1000.0
            )
            self.battery.apply_calendar_aging(days=1.0, temp_c=self.battery.current_temp_c)
            if daily_throughput_kwh > 0:
                self.battery.apply_cycle_aging(
                    energy_cycled_kwh=daily_throughput_kwh,
                    temp_c=self.battery.current_temp_c
                )
            self.battery.complete_cycle()
            self._last_battery_throughput_mwh = self.battery.total_throughput_mwh
    
    def _calculate_performance_metrics(self, results_df):
        """Calculate comprehensive performance metrics."""
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        
        metrics = {}
        
        # Energy metrics
        metrics['total_solar_generation_kwh'] = results_df['solar_generation_kw'].sum() * self.timestep_hours
        metrics['total_wind_generation_kwh'] = results_df['wind_generation_kw'].sum() * self.timestep_hours
        metrics['total_hydro_generation_kwh'] = results_df['hydro_generation_kw'].sum() * self.timestep_hours
        metrics['total_diesel_generation_kwh'] = results_df['diesel_generation_kw'].sum() * self.timestep_hours
        metrics['total_battery_charge_kwh'] = results_df['battery_charge_kw'].sum() * self.timestep_hours
        metrics['total_battery_discharge_kwh'] = results_df['battery_discharge_kw'].sum() * self.timestep_hours
        metrics['total_battery_throughput_kwh'] = self.battery.total_throughput_mwh * 1000.0
        metrics['total_generation_kwh'] = results_df['total_generation_kw'].sum() * self.timestep_hours
        metrics['total_supply_kwh'] = results_df['total_supply_kw'].sum() * self.timestep_hours
        
        metrics['total_baseline_load_kwh'] = results_df['baseline_load_kw'].sum() * self.timestep_hours
        metrics['total_load_kwh'] = results_df['load_kw'].sum() * self.timestep_hours
        metrics['total_load_served_kwh'] = results_df['load_served_kw'].sum() * self.timestep_hours
        metrics['total_load_shedding_kwh'] = results_df['load_shedding_kw'].sum() * self.timestep_hours
        metrics['total_dsm_shifted_energy_kwh'] = results_df['dsm_shifted_load_kw'].clip(lower=0.0).sum() * self.timestep_hours
        metrics['total_peak_reduced_energy_kwh'] = results_df['dsm_peak_reduction_kw'].sum() * self.timestep_hours
        metrics['total_direct_renewable_served_kwh'] = results_df['renewable_served_kw'].sum() * self.timestep_hours
        metrics['total_renewable_from_battery_served_kwh'] = results_df['renewable_battery_served_kw'].sum() * self.timestep_hours
        metrics['total_renewable_served_kwh'] = results_df['total_renewable_served_kw'].sum() * self.timestep_hours
        metrics['total_battery_served_kwh'] = results_df['battery_served_kw'].sum() * self.timestep_hours
        metrics['total_diesel_served_kwh'] = results_df['diesel_served_kw'].sum() * self.timestep_hours
        metrics['simulated_years'] = self.simulated_years

        # Reliability metrics
        metrics['loss_of_load_hours'] = len(results_df[results_df['load_shedding_kw'] > 0.1]) * self.timestep_hours
        metrics['loss_of_load_probability'] = metrics['loss_of_load_hours'] / (
            len(results_df) * self.timestep_hours + 1e-6
        )
        metrics['diesel_unavailable_hours'] = (~results_df['diesel_available'].astype(bool)).sum() * self.timestep_hours
        metrics['diesel_availability_fraction'] = 1.0 - (
            metrics['diesel_unavailable_hours'] / (len(results_df) * self.timestep_hours + 1e-6)
        )
        metrics['diesel_forced_outage_events'] = self.diesel_forced_outage_events
        metrics['diesel_planned_outage_events'] = self.diesel_planned_outage_events
        metrics['load_served_fraction'] = (metrics['total_load_served_kwh'] / 
                                          (metrics['total_load_kwh'] + 1e-6))
        
        # Renewable metrics
        total_renewable = (metrics['total_solar_generation_kwh'] + 
                          metrics['total_wind_generation_kwh'] +
                          metrics['total_hydro_generation_kwh'])
        metrics['direct_renewable_fraction'] = (
            metrics['total_direct_renewable_served_kwh'] / (metrics['total_load_served_kwh'] + 1e-6)
        )
        metrics['renewable_fraction'] = (
            metrics['total_renewable_served_kwh'] / (metrics['total_load_served_kwh'] + 1e-6)
        )
        
        # Efficiency metrics
        metrics['average_battery_soc'] = results_df['battery_soc_after'].mean()
        metrics['min_battery_soc'] = results_df['battery_soc_after'].min()
        metrics['max_battery_soc'] = results_df['battery_soc_after'].max()
        metrics['battery_min_soc_percent'] = self.battery.get_min_soc() * 100.0
        metrics['average_battery_temperature_c'] = results_df['battery_temperature_c'].mean()
        metrics['battery_end_of_life_fraction'] = getattr(self.battery, 'end_of_life_capacity_fraction', 0.80)
        
        # Cost metrics
        metrics['total_operating_cost'] = results_df['operating_cost'].sum()
        metrics['total_fuel_liters'] = results_df['fuel_liters'].sum()
        metrics['diesel_runtime_hours'] = self.diesel_gen.runtime_hours
        metrics['unmet_load_fraction'] = (metrics['total_load_shedding_kwh'] /
                                         (metrics['total_load_kwh'] + 1e-6))
        metrics['operating_cost_per_kwh_served'] = (
            metrics['total_operating_cost'] / (metrics['total_load_served_kwh'] + 1e-6)
        )

        # Average power
        metrics['average_load_kw'] = results_df['load_kw'].mean()
        metrics['peak_baseline_load_kw'] = results_df['baseline_load_kw'].max()
        metrics['peak_load_kw'] = results_df['load_kw'].max()
        metrics['average_solar_kw'] = results_df['solar_generation_kw'].mean()
        metrics['average_wind_kw'] = results_df['wind_generation_kw'].mean()
        metrics['average_hydro_kw'] = results_df['hydro_generation_kw'].mean()

        annual_metrics = {}
        for key in [
            'total_solar_generation_kwh',
            'total_wind_generation_kwh',
            'total_hydro_generation_kwh',
            'total_battery_charge_kwh',
            'total_diesel_generation_kwh',
            'total_battery_discharge_kwh',
            'total_battery_throughput_kwh',
            'total_generation_kwh',
            'total_supply_kwh',
            'total_load_kwh',
            'total_load_served_kwh',
            'total_load_shedding_kwh',
            'total_renewable_served_kwh',
            'total_battery_served_kwh',
            'total_diesel_served_kwh',
            'total_operating_cost',
            'total_fuel_liters',
            'diesel_runtime_hours',
        ]:
            annual_metrics[f'annual_{key}'] = self._annualize_value(metrics[key])

        metrics.update(annual_metrics)
        metrics['annual_average_battery_temperature_c'] = metrics['average_battery_temperature_c']

        lifecycle_cashflow, lifecycle_metrics = self._build_lifecycle_cashflow({
            'total_load_served_kwh': metrics['annual_total_load_served_kwh'],
            'total_load_shedding_kwh': metrics['annual_total_load_shedding_kwh'],
            'total_fuel_liters': metrics['annual_total_fuel_liters'],
            'total_diesel_generation_kwh': metrics['annual_total_diesel_generation_kwh'],
            'total_battery_discharge_kwh': metrics['annual_total_battery_discharge_kwh'],
            'total_battery_throughput_kwh': metrics['annual_total_battery_throughput_kwh'],
            'diesel_runtime_hours': metrics['annual_diesel_runtime_hours'],
            'average_battery_temperature_c': metrics['annual_average_battery_temperature_c'],
        })
        self.economic_cashflow = lifecycle_cashflow

        monte_carlo_samples, monte_carlo_summary = self._run_monte_carlo_financial_risk({
            'total_load_served_kwh': metrics['annual_total_load_served_kwh'],
            'total_load_shedding_kwh': metrics['annual_total_load_shedding_kwh'],
            'total_fuel_liters': metrics['annual_total_fuel_liters'],
            'total_diesel_generation_kwh': metrics['annual_total_diesel_generation_kwh'],
            'total_battery_discharge_kwh': metrics['annual_total_battery_discharge_kwh'],
            'total_battery_throughput_kwh': metrics['annual_total_battery_throughput_kwh'],
            'diesel_runtime_hours': metrics['annual_diesel_runtime_hours'],
            'average_battery_temperature_c': metrics['annual_average_battery_temperature_c'],
        })
        self.monte_carlo_samples = monte_carlo_samples
        self.monte_carlo_summary = monte_carlo_summary

        metrics['upfront_capex'] = lifecycle_metrics['upfront_capex']
        metrics['discounted_lifecycle_cost'] = lifecycle_metrics['discounted_lifecycle_cost']
        metrics['discounted_operating_cost'] = lifecycle_metrics['discounted_operating_cost']
        metrics['discounted_unserved_energy_cost'] = lifecycle_metrics['discounted_unserved_energy_cost']
        metrics['discounted_salvage_value'] = lifecycle_metrics['discounted_salvage_value']
        metrics['discounted_energy_served_kwh'] = lifecycle_metrics['discounted_energy_served_kwh']
        metrics['lcoe'] = lifecycle_metrics['lcoe']
        metrics['cost_per_kwh_served'] = lifecycle_metrics['lcoe']
        metrics['project_lifetime_years'] = lifecycle_metrics['project_lifetime_years']
        metrics['annual_fixed_om_base'] = lifecycle_metrics['annual_fixed_om_base']
        metrics['battery_replacement_interval_years'] = lifecycle_metrics['battery_replacement_interval_years']
        metrics['annual_debt_service'] = lifecycle_metrics['annual_debt_service']
        metrics['average_dscr'] = lifecycle_metrics['average_dscr']
        metrics['minimum_dscr'] = lifecycle_metrics['minimum_dscr']
        if not monte_carlo_summary.empty:
            lcoe_row = monte_carlo_summary.loc[monte_carlo_summary['metric'] == 'lcoe'].iloc[0]
            dscr_row = monte_carlo_summary.loc[monte_carlo_summary['metric'] == 'minimum_dscr'].iloc[0]
            npv_row = monte_carlo_summary.loc[monte_carlo_summary['metric'] == 'discounted_lifecycle_cost'].iloc[0]
            metrics['monte_carlo_runs'] = len(monte_carlo_samples)
            metrics['lcoe_p50'] = lcoe_row['p50']
            metrics['lcoe_p90'] = lcoe_row['p90']
            metrics['discounted_lifecycle_cost_p50'] = npv_row['p50']
            metrics['discounted_lifecycle_cost_p90'] = npv_row['p90']
            metrics['minimum_dscr_p10'] = dscr_row['p10']
        else:
            metrics['monte_carlo_runs'] = 0
            metrics['lcoe_p50'] = np.nan
            metrics['lcoe_p90'] = np.nan
            metrics['discounted_lifecycle_cost_p50'] = np.nan
            metrics['discounted_lifecycle_cost_p90'] = np.nan
            metrics['minimum_dscr_p10'] = np.nan
        
        # Print metrics
        print(f"\nGENERATION SUMMARY:")
        print(f"  Total Solar:          {metrics['total_solar_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Wind:           {metrics['total_wind_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Hydro:          {metrics['total_hydro_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Diesel:         {metrics['total_diesel_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Generation:     {metrics['total_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Supply:         {metrics['total_supply_kwh']:>12,.1f} kWh")
        
        print(f"\nLOAD & RELIABILITY:")
        print(f"  Total Load Demand:    {metrics['total_load_kwh']:>12,.1f} kWh")
        print(f"  Total Load Served:    {metrics['total_load_served_kwh']:>12,.1f} kWh")
        print(f"  Load Shedding:        {metrics['total_load_shedding_kwh']:>12,.1f} kWh")
        print(f"  Served Fraction:      {metrics['load_served_fraction']:>12.2%}")
        print(f"  Loss of Load Hours:   {metrics['loss_of_load_hours']:>12.0f} hrs")
        print(f"  Loss of Load Prob:    {metrics['loss_of_load_probability']:>12.2%}")
        
        print(f"\nRENEWABLE PENETRATION:")
        print(f"  Gross Renewable:      {total_renewable:>12,.1f} kWh")
        print(f"  Renewable Served:     {metrics['total_renewable_served_kwh']:>12,.1f} kWh")
        print(f"  Direct Renewable:     {metrics['total_direct_renewable_served_kwh']:>12,.1f} kWh")
        print(f"  Renewable via Battery:{metrics['total_renewable_from_battery_served_kwh']:>12,.1f} kWh")
        print(f"  Direct Ren. Fraction: {metrics['direct_renewable_fraction']:>12.2%}")
        print(f"  Renewable Fraction:   {metrics['renewable_fraction']:>12.2%}")
        
        print(f"\nBATTERY PERFORMANCE:")
        print(f"  Average SOC:          {metrics['average_battery_soc']:>12.1f} %")
        print(f"  Min SOC:              {metrics['min_battery_soc']:>12.1f} %")
        print(f"  Max SOC:              {metrics['max_battery_soc']:>12.1f} %")
        
        print(f"\nECONOMIC PERFORMANCE:")
        print(f"  Total Op. Cost:       ${metrics['total_operating_cost']:>12,.2f}")
        print(f"  Op. Cost / kWh:       ${metrics['operating_cost_per_kwh_served']:>12,.3f}/kWh")
        print(f"  Upfront CAPEX:        ${metrics['upfront_capex']:>12,.2f}")
        print(f"  Lifecycle Cost (NPV): ${metrics['discounted_lifecycle_cost']:>12,.2f}")
        print(f"  LCOE:                 ${metrics['lcoe']:>12,.3f}/kWh")
        print(f"  Total Fuel Used:      {metrics['total_fuel_liters']:>12,.2f} L")
        print(f"  Diesel Runtime:       {metrics['diesel_runtime_hours']:>12,.1f} h")

        print(f"\nRELIABILITY:")
        print(f"  Unmet Load %:         {metrics['unmet_load_fraction']:>12.2%}")
        print(f"  Loss of Load Hours:   {metrics['loss_of_load_hours']:>12.0f} hrs")
        print(f"  Loss of Load Prob:    {metrics['loss_of_load_probability']:>12.2%}")

        print(f"\nLOAD CHARACTERISTICS:")
        print(f"  Average Load:         {metrics['average_load_kw']:>12.1f} kW")
        print(f"  Peak Load:            {metrics['peak_load_kw']:>12.1f} kW")
        
        print()
        
        self.performance_metrics = metrics
        return metrics
    
    def _save_results(self, results_df):
        """Save simulation results to CSV file."""
        filename = (f"microgrid_results_{self.timestep_minutes}min_"
                   f"{self.num_days}days_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        filepath = Path(filename)
        
        results_df.to_csv(filepath, index=False)
        print(f"+ Results saved to: {filepath}")
        
        # Also save a summary metrics file
        metrics_filename = (f"microgrid_metrics_{self.timestep_minutes}min_"
                           f"{self.num_days}days_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"+ Metrics saved to: {metrics_filename}")

        if not self.economic_cashflow.empty:
            cashflow_filename = (f"microgrid_cashflow_{self.timestep_minutes}min_"
                                f"{self.num_days}days_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.economic_cashflow.to_csv(cashflow_filename, index=False)
            print(f"+ Lifecycle cash flow saved to: {cashflow_filename}")
    
    def plot_results(self, days_to_plot=7, save_figure=True):
        """
        Create comprehensive visualization of simulation results.
        
        Args:
            days_to_plot (int): Number of days to plot (from start of simulation)
            save_figure (bool): Whether to save figure to file
        """
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Select timeframe
        timesteps_to_plot = min(days_to_plot * self.steps_per_day, len(results_df))
        plot_data = results_df.iloc[:timesteps_to_plot].copy()
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        fig.suptitle(f'Microgrid Performance - {self.timestep_minutes}min Timestep',
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Power generation and demand
        ax = axes[0]
        ax.plot(plot_data.index, plot_data['solar_generation_kw'], label='Solar', linewidth=1.5)
        ax.plot(plot_data.index, plot_data['wind_generation_kw'], label='Wind', linewidth=1.5)
        ax.plot(plot_data.index, plot_data['hydro_generation_kw'], label='Hydro', linewidth=1.5)
        ax.plot(plot_data.index, plot_data['diesel_generation_kw'], label='Diesel', linewidth=1.5)
        ax.plot(plot_data.index, plot_data['load_kw'], label='Load', 
               linewidth=2, linestyle='--', color='black')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Power Generation vs Load')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Battery state of charge
        ax = axes[1]
        ax.fill_between(plot_data.index, 0, plot_data['battery_soc_after'], 
                       alpha=0.6, label='Battery SOC')
        min_soc_percent = self.battery.get_min_soc() * 100.0
        ax.axhline(
            y=min_soc_percent,
            color='red',
            linestyle='--',
            label=f'Min SOC ({min_soc_percent:.0f}%)',
            linewidth=1
        )
        ax.axhline(y=100, color='green', linestyle='--', label='Max SOC (100%)', linewidth=1)
        ax.set_ylabel('State of Charge (%)')
        ax.set_title('Battery State of Charge')
        ax.set_ylim([0, 105])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Generation mix stacked area
        ax = axes[2]
        ax.stackplot(plot_data.index,
                    plot_data['solar_generation_kw'],
                    plot_data['wind_generation_kw'],
                    plot_data['hydro_generation_kw'],
                    plot_data['diesel_generation_kw'],
                    plot_data['battery_discharge_kw'],
                    labels=['Solar', 'Wind', 'Hydro', 'Diesel', 'Battery'],
                    alpha=0.8)
        ax.plot(plot_data.index, plot_data['load_kw'], 'k--', linewidth=2, label='Load')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Generation Mix (Stacked)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Load shedding and reserve margin
        ax = axes[3]
        ax.bar(plot_data.index, plot_data['load_shedding_kw'], 
              label='Load Shedding', color='red', alpha=0.7)
        ax.set_ylabel('Load Shedding (kW)')
        ax.set_xlabel('Time (hours)')
        ax.set_title('Load Shedding Events')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_figure:
            fig_filename = (f"microgrid_plot_{self.timestep_minutes}min_"
                          f"{days_to_plot}days_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
            print(f"+ Figure saved to: {fig_filename}")
        
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Generic example configuration for a one-year hybrid microgrid run.
    # Design parameters:
    #   - Diesel generator: 60 kVA at 0.8 power factor (≈48 kW prime)
    #   - PV array: 100 kWp
    #   - Battery: 200 kWh (power 60 kW)
    #   - Dispatch: load following by default; switch to 'cycle_charging' if needed

    print("\n" + "="*70)
    print("HYBRID MINI-GRID SIMULATION (1 YEAR)")
    print("="*70 + "\n")

    sim = MicrogridSimulation(
        timestep_minutes=60,
        num_days=365,
        start_date='2026-01-01',
        pv_capacity_kwp=100.0,
        wind_capacity_kw=0.0,  # No wind for this design case
        diesel_capacity_kva=60.0,
        diesel_power_factor=0.8,
        battery_capacity_kwh=200.0,
        battery_power_kw=60.0,
        base_load_kw=40.0,
        load_type='residential',
        dispatch_strategy='load_following'
    )

    results = sim.run_simulation(save_results=True, verbose=False)
    sim.plot_results(days_to_plot=14, save_figure=True)


    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
