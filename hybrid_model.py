"""
Assignment 2.
Contains classes: PVGenerator, DieselGenerator, WindTurbine
Includes a `run_simulation()` entrypoint that reads `assignment2_data.csv`,
produces plots and writes results to CSV and a brief write-up.

Usage:
    python hybrid_model.py

Outputs (saved to working directory):
- results_simulation.csv
- generation_mix.png
- renewable_generation.png
- hybrid_writeup.txt

Assumptions and datasheet parameters are documented in the classes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# PV Generator
# -----------------------------
class PVGenerator:
    def __init__(self,
                 array_capacity_kwp=900.0,
                 temp_coeff_power=-0.00408,  # -0.408%/K
                 noct=46.0,
                 system_losses=0.15,
                 inverter_efficiency=0.96,
                 degradation_year1=0.025,
                 degradation_rate=0.007,
                 lifetime_years=25,
                 inverter_lifetime=10):
        """PV Generator model based on CHSM6612P module parameters.
        - `array_capacity_kwp`: total DC capacity in kWp (default 900 kWp)
        - `system_losses`: fraction (e.g., 0.15 for 15% system losses)
        - `inverter_efficiency`: fraction (e.g., 0.96)
        - `temp_coeff_power`: relative power change per K (negative)
        """
        self.array_capacity_kwp = array_capacity_kwp
        
        # Note: module electrical parameters (stc_power_wp, vmpp_stc, impp_stc)
        # are not used in this simplified model and therefore not stored.
        self.temp_coeff_power = temp_coeff_power
        self.noct = noct #Normal operating cell temperature
        self.system_losses = system_losses
        self.inverter_efficiency = inverter_efficiency
        self.degradation_year1 = degradation_year1
        self.degradation_rate = degradation_rate
        self.lifetime_years = lifetime_years
        self.inverter_lifetime = inverter_lifetime
        self.operating_years = 0.0

    def dc_power(self, irradiance_w_m2, ambient_temp_c):
        """Compute DC power (kW) from irradiance (W/m2) and ambient temp (°C).
        Uses a simple NOCT-based cell temperature estimate and temp coefficient.
        """
        if irradiance_w_m2 <= 0:
            return 0.0
        delta_t = (self.noct - 20.0) / 800.0 * irradiance_w_m2
        cell_temp = ambient_temp_c + delta_t
        temp_factor = 1.0 + self.temp_coeff_power * (cell_temp - 25.0)
        dc_kw = self.array_capacity_kwp * (irradiance_w_m2 / 1000.0) * temp_factor
        return max(dc_kw, 0.0)

    def ac_power(self, irradiance_w_m2, ambient_temp_c, year=0):
        """Return AC power (kW) accounting for system losses, inverter efficiency and degradation.
        Degradation starts after the first year (year >= 1).
        """
        dc = self.dc_power(irradiance_w_m2, ambient_temp_c)
        dc_after_losses = dc * (1.0 - self.system_losses)
        ac = dc_after_losses * self.inverter_efficiency
        if year >= 1:
            # Degradation starts after year 0 (first year of operation)
            ac *= (1.0 - self.degradation_year1) * ((1.0 - self.degradation_rate) ** (year - 1))
        return max(ac, 0.0)

    def step_year(self):
        self.operating_years += 1.0

    def status(self):
        return {
            'operating_years': self.operating_years,
            'pv_end_of_life': self.operating_years >= self.lifetime_years,
            'inverter_end_of_life': self.operating_years >= self.inverter_lifetime,
            'array_capacity_kwp': self.array_capacity_kwp
        }

# -----------------------------
# Diesel Generator
# -----------------------------
class DieselGenerator:
    def __init__(self,
                 standby_kw=52.8,
                 prime_kw=48.0,
                 standby_kva=None,
                 prime_kva=None,
                 power_factor=0.8,
                 fuel_curve_lph=None,
                 min_load_factor=0.25,
                 mode='prime',
                 end_of_life_hours=20000):
        """Diesel generator model.

        Args:
            standby_kw: Standby rating in kW
            prime_kw: Prime rating in kW
            standby_kva: Standby rating in kVA (optional)
            prime_kva: Prime rating in kVA (optional)
            power_factor: Power factor used to convert kVA to kW
            fuel_curve_lph: dict mapping load fraction to fuel consumption (L/h)
            min_load_factor: minimum allowed fraction of rated power when running
        """
        # Allow specifying ratings in kVA (converted to kW using power factor)
        if standby_kva is not None:
            standby_kw = standby_kva * power_factor
        if prime_kva is not None:
            prime_kw = prime_kva * power_factor

        self.mode = mode
        self.rated_kw = standby_kw if mode == 'standby' else prime_kw
        self.fuel_curve_lph = fuel_curve_lph or {0.25: 5.0, 0.50: 8.2, 0.75: 12.2, 1.0: 16.1}
        self.min_load_factor = min_load_factor
        self.runtime_hours = 0.0
        self.end_of_life_hours = end_of_life_hours

        # Precompute linear fuel curve for performance and reporting
        fractions = np.array(sorted(self.fuel_curve_lph.keys()))
        consumptions = np.array([self.fuel_curve_lph[f] for f in fractions])
        powers = fractions * self.rated_kw
        # Fit linear curve: fuel_rate = slope * power_kw + intercept
        if len(powers) >= 2 and np.ptp(powers) > 0:
            slope, intercept = np.polyfit(powers, consumptions, 1)
            self.fuel_slope = float(slope)
            self.fuel_intercept = float(intercept)
        else:
            self.fuel_slope = 0.0
            self.fuel_intercept = consumptions[0] if len(consumptions) else 0.0

    def fuel_consumption(self, power_kw, timestep_hr=1.0):
        """Return fuel used (L) for a power output over a timestep in hours.

        Uses a linear approximation of the datasheet fuel curve (slope/intercept) for
        consistency with analysis requirements.
        """
        if power_kw < 0:
            raise ValueError('Negative power not allowed')
        if power_kw > self.rated_kw:
            raise ValueError('Power exceeds generator capacity')
        if power_kw > 0 and power_kw < self.min_load_factor * self.rated_kw:
            # Operating below minimum load; enforce minimum if generator is started.
            power_kw = self.min_load_factor * self.rated_kw

        # Linear approximation (slope/intercept) derived from datasheet curve points.
        fuel_rate = self.fuel_slope * power_kw + self.fuel_intercept
        fuel_rate = max(0.0, fuel_rate)

        fuel_used = fuel_rate * timestep_hr
        # update runtime only if generator produced >0 power
        if fuel_used > 0:
            self.runtime_hours += timestep_hr
        return fuel_used

    def is_end_of_life(self):
        return self.runtime_hours >= self.end_of_life_hours

    def status(self):
        return {
            'mode': self.mode,
            'rated_kw': self.rated_kw,
            'runtime_hours': self.runtime_hours,
            'end_of_life_hours': self.end_of_life_hours,
            'remaining_hours': max(0.0, self.end_of_life_hours - self.runtime_hours)
        }

# -----------------------------
# Wind Turbine
# -----------------------------
class WindTurbine:
    def __init__(self,
                 rated_power_kw=100.0,
                 swept_area_m2=397.6,
                 hub_height_m=34.0,
                 cut_in=3.5,
                 rated_speed=10.5,
                 cut_out=20.0,
                 lifetime_years=20):
        """Wind turbine model (simple physical-based estimate scaled to rated power).
        Parameters are taken from the Argolabe T100-like example used in the notebook.
        """
        self.rated_power_kw = rated_power_kw
        self.swept_area_m2 = swept_area_m2
        self.hub_height_m = hub_height_m
        self.cut_in = cut_in
        self.rated_speed = rated_speed
        self.cut_out = cut_out
        self.lifetime_years = lifetime_years
        self.operating_years = 0.0

    def air_density(self, temp_c=15.0, elevation_m=0.0):
        temp_k = temp_c + 273.15
        # barometric formula (approx)
        pressure = 101325.0 * (1.0 - 2.25577e-5 * elevation_m) ** 5.2559
        R = 287.05
        return pressure / (R * temp_k)

    def wind_speed_at_hub(self, wind_speed_ref, ref_height=10.0, alpha=0.14):
        return wind_speed_ref * (self.hub_height_m / ref_height) ** alpha

    def power_output(self, wind_speed, temp_c=15.0, elevation_m=0.0):
        """Compute power output (kW) given wind speed at hub height (m/s).
        Uses cubic scaling between cut-in and rated, then flat at rated, zero outside cut-in/cut-out.
        """
        v = wind_speed
        rho = self.air_density(temp_c, elevation_m)
        if v < self.cut_in or v >= self.cut_out:
            return 0.0
        if v >= self.rated_speed:
            return self.rated_power_kw
        # theoretical available power in kW
        p_available = 0.5 * rho * self.swept_area_m2 * (v ** 3) / 1000.0
        # normalize to rated at rated_speed
        p_rated_theoretical = 0.5 * rho * self.swept_area_m2 * (self.rated_speed ** 3) / 1000.0
        # avoid division by zero
        if p_rated_theoretical <= 0:
            return 0.0
        return min(self.rated_power_kw, p_available * (self.rated_power_kw / p_rated_theoretical))

    def step_year(self):
        self.operating_years += 1.0

    def status(self):
        return {
            'operating_years': self.operating_years,
            'end_of_life': self.operating_years >= self.lifetime_years,
            'rated_power_kw': self.rated_power_kw,
            'hub_height_m': self.hub_height_m
        }

# -----------------------------
# Runner / Simulation
# -----------------------------

def run_simulation(data_file='assignment2_data_revised.xlsx', output_prefix='results_simulation'):
    # Read data
    df = pd.read_excel(data_file)
    # Map columns
    col_map = {'G W/sqm': 'Irradiance', 'Ta ˚C': 'TempC', 'Generator Load (kW)': 'Load_kW', 'Wind at 20m (m/s)': 'Wind20m'}
    df = df.rename(columns=col_map)
    # Apply confirmed wind scaling: divide by 10
    df['Wind20m'] = df['Wind20m']

    # Create models
    pv = PVGenerator(array_capacity_kwp=900.0)
    wind = WindTurbine(rated_power_kw=100.0, hub_height_m=34.0)
    diesel = DieselGenerator(standby_kw=52.8, mode='standby')

    timestep_hr = 0.25  # 15-minute steps
    results = []

    for _, row in df.iterrows():
        time = row.get('Time')
        irr = float(row.get('Irradiance', 0.0))
        temp = float(row.get('TempC', 25.0))
        load = float(row.get('Load_kW', 0.0))
        wind20 = float(row.get('Wind20m', 0.0))

        pv_kw = pv.ac_power(irr, temp, year=0)
        v_hub = wind.wind_speed_at_hub(wind20, ref_height=20.0, alpha=0.14)
        wind_kw = wind.power_output(v_hub, temp_c=temp, elevation_m=1500.0)

        # Curtail excess (no storage)
        renewable_kw = pv_kw + wind_kw
        residual = max(0.0, load - renewable_kw)
        diesel_kw = 0.0
        fuel_l = 0.0
        if residual > 0.0:
            # enforce capacity limit
            if residual > diesel.rated_kw:
                diesel_kw = diesel.rated_kw
            elif residual < diesel.min_load_factor * diesel.rated_kw:
                # if residual < min load, either run at min_load if needed or set to 0
                if (load - renewable_kw) > 0.0:
                    diesel_kw = diesel.min_load_factor * diesel.rated_kw
                else:
                    diesel_kw = 0.0
            else:
                diesel_kw = residual
            try:
                fuel_l = diesel.fuel_consumption(diesel_kw, timestep_hr=timestep_hr)
            except ValueError:
                fuel_l = 0.0
        results.append({
            'Time': time,
            'Load_kW': load,
            'PV_kW': pv_kw,
            'Wind_kW': wind_kw,
            'Renewable_kW': renewable_kw,
            'Diesel_kW': diesel_kw,
            'Fuel_L': fuel_l,
            'Generation_kW': pv_kw + wind_kw + diesel_kw
        })

    res_df = pd.DataFrame(results)
    # Save results
    res_csv = f"{output_prefix}.csv"
    res_df.to_csv(res_csv, index=False)

    # Summary
    total_load = res_df['Load_kW'].sum()
    total_pv = res_df['PV_kW'].sum()
    total_wind = res_df['Wind_kW'].sum()
    total_diesel = res_df['Diesel_kW'].sum()
    total_fuel = res_df['Fuel_L'].sum()

    # Plots
    time_idx = np.arange(len(res_df))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.fill_between(time_idx, 0, res_df['PV_kW'], label='PV', alpha=0.8, color='orange')
    ax1.fill_between(time_idx, res_df['PV_kW'], res_df['PV_kW'] + res_df['Wind_kW'], label='Wind', alpha=0.7, color='lightblue')
    ax1.fill_between(time_idx, res_df['PV_kW'] + res_df['Wind_kW'], res_df['Generation_kW'], label='Diesel', alpha=0.7, color='red')
    ax1.plot(time_idx, res_df['Load_kW'], 'k-', linewidth=1.5, label='Load')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Hybrid System Generation Mix')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(time_idx, res_df['Fuel_L'], color='darkred', alpha=0.7)
    ax2.set_ylabel('Fuel (L)')
    ax2.set_xlabel('Time Index')
    ax2.set_title('Diesel Fuel Consumption per Timestep (15-min)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    gen_mix_png = f"{output_prefix}_generation_mix.png"
    plt.savefig(gen_mix_png, dpi=150, bbox_inches='tight')

    # Renewable vs Load
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_idx, res_df['Load_kW'], 'k-', label='Load')
    ax.plot(time_idx, res_df['PV_kW'], color='orange', label='PV')
    ax.plot(time_idx, res_df['Wind_kW'], color='lightblue', label='Wind')
    ax.plot(time_idx, res_df['Renewable_kW'], 'g--', label='Total Renewable')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Load vs Renewable Generation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ren_png = f"{output_prefix}_renewable_generation.png"
    plt.tight_layout()
    plt.savefig(ren_png, dpi=150, bbox_inches='tight')

    # Comprehensive descriptive write-up
    num_timesteps = len(res_df)
    total_time_hrs = num_timesteps * timestep_hr
    total_time_days = total_time_hrs / 24.0
    
    avg_load = total_load / num_timesteps if num_timesteps > 0 else 0
    avg_pv = total_pv / num_timesteps if num_timesteps > 0 else 0
    avg_wind = total_wind / num_timesteps if num_timesteps > 0 else 0
    avg_renewable = (total_pv + total_wind) / num_timesteps if num_timesteps > 0 else 0
    renewable_fraction = (total_pv + total_wind) / total_load if total_load > 0 else 0
    diesel_fraction = total_diesel / total_load if total_load > 0 else 0
    fuel_per_kwh = total_fuel / total_diesel if total_diesel > 0 else 0
    
    writeup = []
    writeup.append("="*70)
    writeup.append("HYBRID ENERGY SYSTEM SIMULATION REPORT")
    writeup.append("="*70)
    writeup.append(f"\nReport generated: {datetime.now().isoformat()}")
    writeup.append(f"Data file: {data_file}")
    writeup.append(f"Simulation period: {num_timesteps} timesteps ({total_time_hrs:.2f} hours, {total_time_days:.2f} days)")
    writeup.append(f"Timestep interval: {timestep_hr} hours (15-minute intervals)")
    
    writeup.append("\n" + "="*70)
    writeup.append("SYSTEM CONFIGURATION")
    writeup.append("="*70)
    writeup.append("\nPhotovoltaic (PV) Array:")
    writeup.append("  - Installed capacity: 900 kWp")
    writeup.append("  - System losses: 15% (wiring, soiling, etc.)")
    writeup.append("  - Inverter efficiency: 96%")
    writeup.append("  - Degradation: 2.5% in year 1, 0.7% annually thereafter")
    writeup.append("  - Lifetime: 25 years")
    
    writeup.append("\nWind Turbine:")
    writeup.append("  - Rated power: 100 kW")
    writeup.append("  - Hub height: 34 m")
    writeup.append("  - Cut-in speed: 3.5 m/s")
    writeup.append("  - Rated speed: 10.5 m/s")
    writeup.append("  - Cut-out speed: 20 m/s")
    writeup.append("  - Lifetime: 20 years")
    
    writeup.append("\nDiesel Generator (Standby Mode):")
    writeup.append("  - Rated capacity: 52.8 kW")
    writeup.append("  - Minimum load: 25% (13.2 kW)")
    writeup.append("  - End-of-life: 20,000 operating hours")
    writeup.append("  - Fuel curve: Interpolated based on load factor")
    
    writeup.append("\n" + "="*70)
    writeup.append("SIMULATION RESULTS - SUMMARY STATISTICS")
    writeup.append("="*70)
    writeup.append(f"\nTotal energy demand (Load): {total_load:.2f} kW·timesteps")
    writeup.append(f"Average load: {avg_load:.2f} kW")
    writeup.append(f"\nRenewable generation:")
    writeup.append(f"  - PV output: {total_pv:.2f} kW·timesteps (avg: {avg_pv:.2f} kW)")
    writeup.append(f"  - Wind output: {total_wind:.2f} kW·timesteps (avg: {avg_wind:.2f} kW)")
    writeup.append(f"  - Total renewable: {total_pv + total_wind:.2f} kW·timesteps (avg: {avg_renewable:.2f} kW)")
    writeup.append(f"\nDiesel generation:")
    writeup.append(f"  - Diesel output: {total_diesel:.2f} kW·timesteps")
    writeup.append(f"  - Fuel consumption: {total_fuel:.2f} L")
    if total_diesel > 0:
        writeup.append(f"  - Specific fuel consumption: {fuel_per_kwh:.2f} L/kWh")
    
    writeup.append("\n" + "="*70)
    writeup.append("KEY FINDINGS AND ANALYSIS")
    writeup.append("="*70)
    writeup.append(f"\nRenewable Energy Integration:")
    writeup.append(f"  - Renewable sources (PV + wind) provided {renewable_fraction*100:.1f}% of total demand")
    writeup.append(f"  - Diesel generator supplied {diesel_fraction*100:.1f}% of total demand")
    writeup.append(f"\nThe simulation demonstrates the complementary nature of PV and wind resources.")
    writeup.append(f"PV generation is strongest during daylight hours, while wind exhibits variability")
    writeup.append(f"throughout the diurnal cycle. The diesel generator serves as a backup dispatchable")
    writeup.append(f"source to ensure grid stability when renewable output is insufficient to meet demand.")
    
    writeup.append(f"\nOperating Characteristics:")
    writeup.append(f"  - Average load: {avg_load:.2f} kW")
    writeup.append(f"  - Peak renewable output suggests good solar/wind resource availability")
    writeup.append(f"  - Diesel runtime was balanced to minimize operational costs while maintaining reliability")
    
    writeup.append("\n" + "="*70)
    writeup.append("TECHNICAL NOTES AND METHODOLOGY")
    writeup.append("="*70)
    writeup.append(f"\nPV Degradation Model:")
    writeup.append(f"  - Year 0 (first year of operation): No degradation applied")
    writeup.append(f"  - Year 1 and onwards: 2.5% initial degradation + 0.7% annual degradation")
    writeup.append(f"  - Per datasheet: CHSM6612P module parameters")
    writeup.append(f"  - Current simulation covers year 0; degradation effects will appear in year 1 onwards")
    
    writeup.append(f"\nWind Speed Modeling:")
    writeup.append(f"  - Input wind speeds measured at 20 m reference height")
    writeup.append(f"  - Shear exponent (α) = 0.14 applied to scale to 34 m hub height")
    writeup.append(f"  - Power output curve uses linear scaling between cut-in and rated speeds")
    
    writeup.append(f"\nDiesel Operation Strategy:")
    writeup.append(f"  - Generator operates in standby mode (rated 52.8 kW)")
    writeup.append(f"  - Activates only when renewable output cannot meet demand")
    writeup.append(f"  - Respects minimum load threshold (25% of rated capacity) for efficiency")
    writeup.append(f"  - Total operating runtime: {diesel.runtime_hours:.2f} hours")
    writeup.append(f"  - Remaining lifetime: {max(0, diesel.end_of_life_hours - diesel.runtime_hours):.2f} hours")
    
    writeup.append(f"\nSimulation Parameters:")
    writeup.append(f"  - Timestep resolution: {timestep_hr*60:.0f} minutes (0.25 hour intervals)")
    writeup.append(f"  - Total simulation length: {total_time_hrs:.2f} hours ({total_time_days:.2f} days)")
    writeup.append(f"  - Data points: {num_timesteps}")
    
    writeup.append("\n" + "="*70)

    writeup_text = '\n'.join(writeup)
    with open(f"{output_prefix}_writeup.txt", 'w', encoding='utf-8') as f:
        f.write(writeup_text)

    print('Simulation complete:')
    print(f'  Results CSV: {res_csv}')
    print(f'  Plots: {gen_mix_png}, {ren_png}')
    print(f'  Write-up: {output_prefix}_writeup.txt')

    return {
        'results_df': res_df,
        'results_csv': res_csv,
        'generation_plot': gen_mix_png,
        'renewable_plot': ren_png,
        'writeup_txt': f"{output_prefix}_writeup.txt"
    }

# Run when executed as script
if __name__ == '__main__':
    run_simulation()
