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

from solar_pv_model import pv_power_from_ghi
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

        pv_config = {'array_capacity_kwp': pv_capacity_kwp}
        if pv_params:
            pv_config.update(pv_params)
        self.pv_gen = PVGenerator(**pv_config)

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
            diesel_config.update(diesel_params)
        self.diesel_gen = DieselGenerator(**diesel_config)
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
            load_config.update(load_params)
        self.load_model = MicrogridLoad(**load_config)
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
        print(f"  + Project life: {self.economic_params['project_lifetime_years']} years")
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

    def _build_economic_params(self, overrides=None):
        """Build lifecycle economics assumptions with user overrides."""
        defaults = {
            'currency': 'USD',
            'project_lifetime_years': 20,
            'nominal_discount_rate': 0.12,
            'general_inflation_rate': 0.03,
            'fuel_price_per_liter': 1.50,
            'fuel_price_escalation_rate': 0.05,
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
        }
        if overrides:
            defaults.update(overrides)
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

    def _estimate_battery_replacement_interval_years(self, annual_battery_throughput_kwh):
        """Estimate battery life from calendar and throughput constraints."""
        calendar_life = max(self.battery.lifetime_years, 1.0)
        throughput_limit_mwh = getattr(self.battery, 'lifetime_throughput_MWh', None)
        if throughput_limit_mwh is not None and annual_battery_throughput_kwh > 0:
            throughput_life = (throughput_limit_mwh * 1000.0) / annual_battery_throughput_kwh
            calendar_life = min(calendar_life, max(throughput_life, 1.0))
        return max(calendar_life, 1.0)

    def _build_lifecycle_cashflow(self, annual_metrics):
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

        annual_battery_throughput_kwh = annual_metrics['total_battery_discharge_kwh']
        battery_replacement_interval = self._estimate_battery_replacement_interval_years(
            annual_battery_throughput_kwh
        )

        annual_runtime = annual_metrics['diesel_runtime_hours']
        diesel_runtime_limit = max(self.diesel_gen.end_of_life_hours, 1.0)

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
                    'battery_variable_om_cost': 0.0,
                    'unserved_energy_cost': 0.0,
                    'replacement_cost': 0.0,
                    'salvage_value': 0.0,
                    'capital_cost': upfront_capex,
                    'total_cost': upfront_capex,
                    'discounted_total_cost': upfront_capex,
                    'discounted_energy_served_kwh': 0.0,
                })
                continue

            fuel_cost = self._escalated_cost(
                annual_metrics['total_fuel_liters'] * econ['fuel_price_per_liter'],
                econ['fuel_price_escalation_rate'],
                year
            )
            fixed_om_cost = self._escalated_cost(
                annual_fixed_om_base,
                econ['om_escalation_rate'],
                year
            )
            diesel_variable_om_cost = self._escalated_cost(
                annual_metrics['total_diesel_generation_kwh'] * econ['diesel_variable_om_per_kwh'],
                econ['om_escalation_rate'],
                year
            )
            battery_variable_om_cost = self._escalated_cost(
                annual_metrics['total_battery_discharge_kwh'] * econ['battery_variable_om_per_kwh'],
                econ['om_escalation_rate'],
                year
            )
            unserved_energy_cost = self._escalated_cost(
                annual_metrics['total_load_shedding_kwh'] * econ['unserved_energy_cost_per_kwh'],
                econ['om_escalation_rate'],
                year
            )

            replacement_cost = 0.0
            cumulative_runtime += annual_runtime

            if annual_runtime > 0 and cumulative_runtime >= diesel_runtime_limit:
                replacement_cost += self._escalated_cost(
                    diesel_capex * econ['diesel_replacement_cost_fraction'],
                    econ['diesel_capex_escalation_rate'],
                    year
                )
                cumulative_runtime = max(0.0, cumulative_runtime - diesel_runtime_limit)
                last_diesel_install_year = year

            if year - last_battery_install_year >= battery_replacement_interval:
                replacement_cost += self._escalated_cost(
                    battery_capex * econ['battery_replacement_cost_fraction'],
                    econ['battery_capex_escalation_rate'],
                    year
                )
                last_battery_install_year = year

            if year - last_pv_install_year >= self.pv_gen.lifetime_years:
                replacement_cost += self._escalated_cost(
                    pv_capex * econ['pv_replacement_cost_fraction'],
                    econ['pv_capex_escalation_rate'],
                    year
                )
                last_pv_install_year = year

            if year - last_wind_install_year >= self.wind_turbine.lifetime_years:
                replacement_cost += self._escalated_cost(
                    wind_capex * econ['wind_replacement_cost_fraction'],
                    econ['wind_capex_escalation_rate'],
                    year
                )
                last_wind_install_year = year

            if year - last_hydro_install_year >= self.hydro_turbine.lifetime_years:
                replacement_cost += self._escalated_cost(
                    hydro_capex * econ['hydro_replacement_cost_fraction'],
                    econ['hydro_capex_escalation_rate'],
                    year
                )
                last_hydro_install_year = year

            if self.pv_gen.inverter_lifetime > 0 and year % self.pv_gen.inverter_lifetime == 0:
                replacement_cost += self._escalated_cost(
                    pv_capex * econ['inverter_replacement_cost_fraction'],
                    econ['pv_capex_escalation_rate'],
                    year
                )

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
                    self._escalated_cost(battery_capex, econ['battery_capex_escalation_rate'], project_life) * battery_remaining_fraction +
                    self._escalated_cost(diesel_capex, econ['diesel_capex_escalation_rate'], project_life) * diesel_remaining_fraction +
                    self._escalated_cost(pv_capex, econ['pv_capex_escalation_rate'], project_life) * pv_remaining_fraction +
                    self._escalated_cost(wind_capex, econ['wind_capex_escalation_rate'], project_life) * wind_remaining_fraction +
                    self._escalated_cost(hydro_capex, econ['hydro_capex_escalation_rate'], project_life) * hydro_remaining_fraction
                )

            annual_total_cost = (
                fixed_om_cost + fuel_cost + diesel_variable_om_cost +
                battery_variable_om_cost + unserved_energy_cost + replacement_cost -
                salvage_value
            )

            discount_factor = self._discount_factor(year)
            discounted_total = annual_total_cost / discount_factor
            discounted_energy = annual_metrics['total_load_served_kwh'] / discount_factor

            discounted_cost_total += discounted_total
            discounted_operating_cost += (
                fixed_om_cost + fuel_cost + diesel_variable_om_cost + battery_variable_om_cost
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
                'battery_variable_om_cost': battery_variable_om_cost,
                'unserved_energy_cost': unserved_energy_cost,
                'replacement_cost': replacement_cost,
                'salvage_value': salvage_value,
                'capital_cost': 0.0,
                'total_cost': annual_total_cost,
                'discounted_total_cost': discounted_total,
                'discounted_energy_served_kwh': discounted_energy,
            })

        cashflow_df = pd.DataFrame(cashflows)
        lcoe = discounted_cost_total / max(discounted_energy_served, 1e-6)
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
        }
        
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

        # Monthly average clearness indices for the default Kigali resource model.
        kigali_monthly_kt = [0.545, 0.562, 0.553, 0.56, 0.559, 0.591,
                             0.581, 0.559, 0.566, 0.545, 0.536, 0.535]

        sim = SolarResourceSimulator()
        hourly_irradiance = np.array(sim.run_full_year(kigali_monthly_kt))

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
            
            # Seasonal variation in wind (typically higher in winter)
            seasonal_mean = 7 + 3 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            
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
        
        day_of_year = np.array([d.dayofyear for d in self.time_index])
        hour_of_day = np.array([d.hour for d in self.time_index])
        
        # Seasonal temperature variation (sine wave, cold in winter, hot in summer)
        seasonal = 15 + 15 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        
        # Daily temperature variation (colder at night, warmer during day)
        daily = 8 * np.sin((hour_of_day - 6) * np.pi / 12)
        
        temperature = seasonal + daily + np.random.normal(0, 2, len(self.time_index))
        
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

        load_values = load_series.values

        print(f"  + Load mean: {np.mean(load_values):.1f} kW")
        print(f"  + Load max: {np.max(load_values):.1f} kW")
        print(f"  + Load min: {np.min(load_values):.1f} kW")
        return load_values

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
                system_size_w=self.pv_gen.array_capacity_kwp * 1000.0
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
            load_kw = load_demand[step]
            
            # Step 3: Run dispatch algorithm (Priority: Solar → Wind → Battery → Diesel)
            dispatch = load_following_algorithm(
                load_kw=load_kw,
                solar_power_kw=solar_kw,
                wind_power_kw=wind_kw,
                hydro_power_kw=hydro_kw,
                battery=self.battery,
                diesel_generator=self.diesel_gen,
                battery_variable_cost_per_kwh=self.economic_params['battery_variable_om_per_kwh'],
                diesel_fuel_price_per_liter=self.economic_params['fuel_price_per_liter'],
                diesel_variable_om_per_kwh=self.economic_params['diesel_variable_om_per_kwh'],
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

            timestep_result = {
                'timestamp': timestamp,
                'step': step,
                'hour_of_year': timestamp.hour + (timestamp.dayofyear - 1) * 24,

                # Demand
                'load_kw': load_kw,
                'load_served_kw': dispatch.get('load_served_kw', 0.0),
                'load_shedding_kw': dispatch.get('load_shedding_kw', 0.0),

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
                'curtailment_kw': dispatch.get('curtailment', 0.0),
                'total_generation_kw': (solar_kw + wind_kw + hydro_kw + dispatch.get('diesel', 0.0) + battery_discharge_kw),

                # Energy balance check
                'power_balance_kw': power_balance_kw,

                # Costs
                'operating_cost': dispatch.get('operating_cost', 0.0),
                'fuel_liters': dispatch.get('fuel_liters', 0.0),
                'fuel_cost': dispatch.get('fuel_liters', 0.0) * self.economic_params['fuel_price_per_liter'],

                # Equipment status
                'pv_operating_years': self.pv_gen.operating_years,
                'diesel_operating_hours': self.diesel_gen.runtime_hours,
                'battery_soc_before': dispatch.get('battery_soc_before', np.nan),
                'battery_soc_after': dispatch.get('battery_soc_after', np.nan),
                'battery_health': self.battery.current_capacity_kwh / self.battery.nominal_capacity_kwh,
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
            self.battery.apply_calendar_aging(days=1.0)
            if daily_throughput_kwh > 0:
                self.battery.apply_cycle_aging(energy_cycled_kwh=daily_throughput_kwh)
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
        metrics['total_battery_discharge_kwh'] = results_df['battery_discharge_kw'].sum() * self.timestep_hours
        metrics['total_generation_kwh'] = results_df['total_generation_kw'].sum() * self.timestep_hours
        
        metrics['total_load_kwh'] = results_df['load_kw'].sum() * self.timestep_hours
        metrics['total_load_served_kwh'] = results_df['load_served_kw'].sum() * self.timestep_hours
        metrics['total_load_shedding_kwh'] = results_df['load_shedding_kw'].sum() * self.timestep_hours
        metrics['simulated_years'] = self.simulated_years

        # Reliability metrics
        metrics['loss_of_load_hours'] = len(results_df[results_df['load_shedding_kw'] > 0.1]) * self.timestep_hours
        metrics['loss_of_load_probability'] = metrics['loss_of_load_hours'] / (
            len(results_df) * self.timestep_hours + 1e-6
        )
        metrics['load_served_fraction'] = (metrics['total_load_served_kwh'] / 
                                          (metrics['total_load_kwh'] + 1e-6))
        
        # Renewable metrics
        total_renewable = (metrics['total_solar_generation_kwh'] + 
                          metrics['total_wind_generation_kwh'] +
                          metrics['total_hydro_generation_kwh'])
        metrics['renewable_fraction'] = min(
            1.0,
            total_renewable / (metrics['total_load_served_kwh'] + 1e-6)
        )
        
        # Efficiency metrics
        metrics['average_battery_soc'] = results_df['battery_soc_after'].mean()
        metrics['min_battery_soc'] = results_df['battery_soc_after'].min()
        metrics['max_battery_soc'] = results_df['battery_soc_after'].max()
        
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
        metrics['peak_load_kw'] = results_df['load_kw'].max()
        metrics['average_solar_kw'] = results_df['solar_generation_kw'].mean()
        metrics['average_wind_kw'] = results_df['wind_generation_kw'].mean()
        metrics['average_hydro_kw'] = results_df['hydro_generation_kw'].mean()

        annual_metrics = {}
        for key in [
            'total_solar_generation_kwh',
            'total_wind_generation_kwh',
            'total_hydro_generation_kwh',
            'total_diesel_generation_kwh',
            'total_battery_discharge_kwh',
            'total_generation_kwh',
            'total_load_kwh',
            'total_load_served_kwh',
            'total_load_shedding_kwh',
            'total_operating_cost',
            'total_fuel_liters',
            'diesel_runtime_hours',
        ]:
            annual_metrics[f'annual_{key}'] = self._annualize_value(metrics[key])

        metrics.update(annual_metrics)

        lifecycle_cashflow, lifecycle_metrics = self._build_lifecycle_cashflow({
            'total_load_served_kwh': metrics['annual_total_load_served_kwh'],
            'total_load_shedding_kwh': metrics['annual_total_load_shedding_kwh'],
            'total_fuel_liters': metrics['annual_total_fuel_liters'],
            'total_diesel_generation_kwh': metrics['annual_total_diesel_generation_kwh'],
            'total_battery_discharge_kwh': metrics['annual_total_battery_discharge_kwh'],
            'diesel_runtime_hours': metrics['annual_diesel_runtime_hours'],
        })
        self.economic_cashflow = lifecycle_cashflow

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
        
        # Print metrics
        print(f"\nGENERATION SUMMARY:")
        print(f"  Total Solar:          {metrics['total_solar_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Wind:           {metrics['total_wind_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Hydro:          {metrics['total_hydro_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Diesel:         {metrics['total_diesel_generation_kwh']:>12,.1f} kWh")
        print(f"  Total Generation:     {metrics['total_generation_kwh']:>12,.1f} kWh")
        
        print(f"\nLOAD & RELIABILITY:")
        print(f"  Total Load Demand:    {metrics['total_load_kwh']:>12,.1f} kWh")
        print(f"  Total Load Served:    {metrics['total_load_served_kwh']:>12,.1f} kWh")
        print(f"  Load Shedding:        {metrics['total_load_shedding_kwh']:>12,.1f} kWh")
        print(f"  Served Fraction:      {metrics['load_served_fraction']:>12.2%}")
        print(f"  Loss of Load Hours:   {metrics['loss_of_load_hours']:>12.0f} hrs")
        print(f"  Loss of Load Prob:    {metrics['loss_of_load_probability']:>12.2%}")
        
        print(f"\nRENEWABLE PENETRATION:")
        print(f"  Total Renewable:      {total_renewable:>12,.1f} kWh")
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
        ax.axhline(y=20, color='red', linestyle='--', label='Min SOC (20%)', linewidth=1)
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
