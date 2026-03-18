"""
Core energy component models for the microgrid simulator.

Currently implemented:
- PVGenerator
- DieselGenerator
- WindTurbine
- HydroTurbine

The project is structured so additional source models can be added over time.
"""

import numpy as np


class PVGenerator:
    def __init__(
        self,
        array_capacity_kwp=900.0,
        temp_coeff_power=-0.00408,
        noct=46.0,
        system_losses=0.15,
        inverter_efficiency=0.96,
        degradation_year1=0.025,
        degradation_rate=0.007,
        lifetime_years=25,
        inverter_lifetime=10,
    ):
        """PV generator model based on a simplified datasheet-driven formulation."""
        self.array_capacity_kwp = array_capacity_kwp
        self.temp_coeff_power = temp_coeff_power
        self.noct = noct
        self.system_losses = system_losses
        self.inverter_efficiency = inverter_efficiency
        self.degradation_year1 = degradation_year1
        self.degradation_rate = degradation_rate
        self.lifetime_years = lifetime_years
        self.inverter_lifetime = inverter_lifetime
        self.operating_years = 0.0

    def dc_power(self, irradiance_w_m2, ambient_temp_c):
        if irradiance_w_m2 <= 0:
            return 0.0
        delta_t = (self.noct - 20.0) / 800.0 * irradiance_w_m2
        cell_temp = ambient_temp_c + delta_t
        temp_factor = 1.0 + self.temp_coeff_power * (cell_temp - 25.0)
        dc_kw = self.array_capacity_kwp * (irradiance_w_m2 / 1000.0) * temp_factor
        return max(dc_kw, 0.0)

    def ac_power(self, irradiance_w_m2, ambient_temp_c, year=0):
        dc = self.dc_power(irradiance_w_m2, ambient_temp_c)
        dc_after_losses = dc * (1.0 - self.system_losses)
        ac = dc_after_losses * self.inverter_efficiency
        if year >= 1:
            ac *= (1.0 - self.degradation_year1) * ((1.0 - self.degradation_rate) ** (year - 1))
        return max(ac, 0.0)

    def step_year(self):
        self.operating_years += 1.0

    def status(self):
        return {
            "operating_years": self.operating_years,
            "pv_end_of_life": self.operating_years >= self.lifetime_years,
            "inverter_end_of_life": self.operating_years >= self.inverter_lifetime,
            "array_capacity_kwp": self.array_capacity_kwp,
        }


class DieselGenerator:
    def __init__(
        self,
        standby_kw=52.8,
        prime_kw=48.0,
        standby_kva=None,
        prime_kva=None,
        power_factor=0.8,
        fuel_curve_lph=None,
        min_load_factor=0.25,
        mode="prime",
        end_of_life_hours=20000,
    ):
        """Diesel generator model driven by rating and fuel-curve inputs."""
        if standby_kva is not None:
            standby_kw = standby_kva * power_factor
        if prime_kva is not None:
            prime_kw = prime_kva * power_factor

        self.mode = mode
        self.rated_kw = standby_kw if mode == "standby" else prime_kw
        self.fuel_curve_lph = fuel_curve_lph or {0.25: 4.50, 0.50: 7.40, 0.75: 11.00, 1.0: 14.70}
        self.min_load_factor = min_load_factor
        self.runtime_hours = 0.0
        self.end_of_life_hours = end_of_life_hours

        fractions = np.array(sorted(self.fuel_curve_lph.keys()))
        consumptions = np.array([self.fuel_curve_lph[f] for f in fractions])
        powers = fractions * self.rated_kw
        if len(powers) >= 2 and np.ptp(powers) > 0:
            slope, intercept = np.polyfit(powers, consumptions, 1)
            self.fuel_slope = float(slope)
            self.fuel_intercept = float(intercept)
        else:
            self.fuel_slope = 0.0
            self.fuel_intercept = consumptions[0] if len(consumptions) else 0.0

    def fuel_consumption(self, power_kw, timestep_hr=1.0):
        if power_kw < 0:
            raise ValueError("Negative power not allowed")
        if power_kw > self.rated_kw:
            raise ValueError("Power exceeds generator capacity")
        if power_kw > 0 and power_kw < self.min_load_factor * self.rated_kw:
            power_kw = self.min_load_factor * self.rated_kw

        fuel_rate = self.fuel_slope * power_kw + self.fuel_intercept
        fuel_rate = max(0.0, fuel_rate)

        fuel_used = fuel_rate * timestep_hr
        if fuel_used > 0:
            self.runtime_hours += timestep_hr
        return fuel_used

    def is_end_of_life(self):
        return self.runtime_hours >= self.end_of_life_hours

    def status(self):
        return {
            "mode": self.mode,
            "rated_kw": self.rated_kw,
            "runtime_hours": self.runtime_hours,
            "end_of_life_hours": self.end_of_life_hours,
            "remaining_hours": max(0.0, self.end_of_life_hours - self.runtime_hours),
        }


class WindTurbine:
    def __init__(
        self,
        rated_power_kw=100.0,
        swept_area_m2=397.6,
        hub_height_m=34.0,
        cut_in=3.5,
        rated_speed=10.5,
        cut_out=20.0,
        lifetime_years=20,
    ):
        """Wind turbine model with a simple datasheet-style power curve."""
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
        pressure = 101325.0 * (1.0 - 2.25577e-5 * elevation_m) ** 5.2559
        gas_constant = 287.05
        return pressure / (gas_constant * temp_k)

    def wind_speed_at_hub(self, wind_speed_ref, ref_height=10.0, alpha=0.14):
        return wind_speed_ref * (self.hub_height_m / ref_height) ** alpha

    def power_output(self, wind_speed, temp_c=15.0, elevation_m=0.0):
        wind = wind_speed
        rho = self.air_density(temp_c, elevation_m)
        if wind < self.cut_in or wind >= self.cut_out:
            return 0.0
        if wind >= self.rated_speed:
            return self.rated_power_kw

        p_available = 0.5 * rho * self.swept_area_m2 * (wind ** 3) / 1000.0
        p_rated_theoretical = 0.5 * rho * self.swept_area_m2 * (self.rated_speed ** 3) / 1000.0
        if p_rated_theoretical <= 0:
            return 0.0
        return min(self.rated_power_kw, p_available * (self.rated_power_kw / p_rated_theoretical))

    def step_year(self):
        self.operating_years += 1.0

    def status(self):
        return {
            "operating_years": self.operating_years,
            "end_of_life": self.operating_years >= self.lifetime_years,
            "rated_power_kw": self.rated_power_kw,
            "hub_height_m": self.hub_height_m,
        }


class HydroTurbine:
    def __init__(
        self,
        rated_power_kw=0.0,
        design_flow_m3s=1.0,
        net_head_m=20.0,
        efficiency=0.85,
        min_flow_fraction=0.2,
        environmental_flow_m3s=0.0,
        lifetime_years=40,
    ):
        """Run-of-river style hydro turbine driven by flow and head."""
        self.rated_power_kw = rated_power_kw
        self.design_flow_m3s = max(design_flow_m3s, 1e-6)
        self.net_head_m = net_head_m
        self.efficiency = efficiency
        self.min_flow_fraction = min_flow_fraction
        self.environmental_flow_m3s = max(environmental_flow_m3s, 0.0)
        self.lifetime_years = lifetime_years
        self.operating_years = 0.0

    def power_output(self, flow_m3s, head_m=None):
        """Compute electrical power from available flow."""
        head = self.net_head_m if head_m is None else head_m
        usable_flow = max(flow_m3s - self.environmental_flow_m3s, 0.0)
        if usable_flow <= 0:
            return 0.0

        min_flow = self.min_flow_fraction * self.design_flow_m3s
        if usable_flow < min_flow:
            return 0.0

        rho = 1000.0
        g = 9.81
        hydraulic_power_kw = rho * g * usable_flow * max(head, 0.0) * self.efficiency / 1000.0
        return min(self.rated_power_kw, hydraulic_power_kw)

    def step_year(self):
        self.operating_years += 1.0

    def status(self):
        return {
            "operating_years": self.operating_years,
            "end_of_life": self.operating_years >= self.lifetime_years,
            "rated_power_kw": self.rated_power_kw,
            "design_flow_m3s": self.design_flow_m3s,
            "net_head_m": self.net_head_m,
        }
