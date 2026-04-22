"""Battery storage model used by the microgrid simulator."""

import numpy as np


class KiBaMBattery:
    """Kinetic Battery Model (KiBaM) based battery storage."""

    def __init__(
        self,
        energy_capacity_kwh=100.0,
        power_capacity_kw=50.0,
        nominal_voltage=48.0,
        charge_efficiency=0.93,
        discharge_efficiency=0.93,
        max_depth_of_discharge=0.8,
        k_rate=0.1,
        c_fraction=0.3,
        temperature_coefficient=-0.003,
        degradation_temp_sensitivity=0.025,
        calendar_fade_rate=0.00005,
        cycle_fade_per_kwh=0.0000003,
        end_of_life_capacity_fraction=0.80,
        lifetime_throughput_MWh=None,
        lifetime_cycles=None,
        lifetime_years=10.0,
    ):
        self.energy_capacity_kwh = energy_capacity_kwh
        self.power_capacity_kw = power_capacity_kw
        self.nominal_voltage = nominal_voltage

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_dod = max_depth_of_discharge
        self.min_soc = 1.0 - max_depth_of_discharge

        self.k_rate = k_rate
        self.c_fraction = c_fraction

        self.capacity_ah = (energy_capacity_kwh * 1000.0) / nominal_voltage
        self.available_ah = self.c_fraction * self.capacity_ah
        self.bound_ah = (1.0 - self.c_fraction) * self.capacity_ah

        self.temperature_coefficient = temperature_coefficient
        self.degradation_temp_sensitivity = degradation_temp_sensitivity
        self.calendar_fade_rate = calendar_fade_rate
        self.cycle_fade_per_kwh = cycle_fade_per_kwh
        self.end_of_life_capacity_fraction = end_of_life_capacity_fraction
        self.nominal_capacity_kwh = energy_capacity_kwh
        self.current_capacity_kwh = energy_capacity_kwh
        self.lifetime_throughput_MWh = lifetime_throughput_MWh
        self.lifetime_cycles = lifetime_cycles
        self.lifetime_years = lifetime_years

        self.total_throughput_mwh = 0.0
        self.cycle_count = 0.0
        self.soc_at_cycle_start = 1.0
        self.throughput_since_cycle_max = 0.0
        self.operating_hours = 0.0
        self.current_temp_c = 25.0

    def _temperature_fade_multiplier(self, temp_c=None):
        if temp_c is None:
            temp_c = self.current_temp_c
        return max(0.25, 1.0 + self.degradation_temp_sensitivity * (temp_c - 25.0))

    def get_energy_kwh(self):
        return (self.available_ah + self.bound_ah) * self.nominal_voltage / 1000.0

    def get_soc(self):
        if self.current_capacity_kwh <= 1e-9:
            return 0.0
        return min(1.0, max(0.0, self.get_energy_kwh() / self.current_capacity_kwh))

    def get_min_soc(self):
        return self.min_soc

    def get_max_charge_power(self, timestep_hr=1.0):
        max_input_power = self.power_capacity_kw
        energy_to_full = self.current_capacity_kwh - self.get_energy_kwh()
        if energy_to_full <= 0:
            return 0.0
        max_input_by_capacity = energy_to_full / max(timestep_hr, 1e-6) / max(self.charge_efficiency, 1e-6)
        return min(max_input_power, max_input_by_capacity)

    def get_max_discharge_power(self, timestep_hr=1.0):
        energy_available = self.get_energy_kwh() - self.min_soc * self.current_capacity_kwh
        if energy_available <= 0:
            return 0.0
        max_output_by_capacity = energy_available * max(self.discharge_efficiency, 1e-6) / max(timestep_hr, 1e-6)
        return min(self.power_capacity_kw, max_output_by_capacity)

    def charge(self, power_kw, timestep_hr=1.0, temp_c=None):
        if power_kw <= 0:
            return 0.0

        if temp_c is not None:
            self.current_temp_c = temp_c

        max_input_power = self.get_max_charge_power(timestep_hr=timestep_hr)
        actual_input_power = min(power_kw, max_input_power)

        energy_drawn_kwh = actual_input_power * timestep_hr
        energy_stored_kwh = energy_drawn_kwh * self.charge_efficiency
        delta_ah = energy_stored_kwh * 1000.0 / max(self.nominal_voltage, 1e-6)

        available_delta = self.c_fraction * delta_ah + self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr
        bound_delta = (1.0 - self.c_fraction) * delta_ah - self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr

        self.available_ah = max(0.0, min(self.capacity_ah, self.available_ah + available_delta))
        self.bound_ah = max(0.0, min(self.capacity_ah - self.available_ah, self.bound_ah + bound_delta))

        self._track_throughput(energy_stored_kwh)
        return energy_drawn_kwh

    def discharge(self, power_kw, timestep_hr=1.0, temp_c=None):
        if power_kw <= 0:
            return 0.0

        if temp_c is not None:
            self.current_temp_c = temp_c

        max_output_power = self.get_max_discharge_power(timestep_hr=timestep_hr)
        actual_output_power = min(power_kw, max_output_power)

        energy_delivered_kwh = actual_output_power * timestep_hr
        energy_removed_kwh = energy_delivered_kwh / max(self.discharge_efficiency, 1e-6)
        delta_ah = energy_removed_kwh * 1000.0 / max(self.nominal_voltage, 1e-6)

        available_delta = -self.c_fraction * delta_ah + self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr
        bound_delta = -(1.0 - self.c_fraction) * delta_ah - self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr

        self.available_ah = max(0.0, self.available_ah + available_delta)
        self.bound_ah = max(0.0, self.bound_ah + bound_delta)

        self._track_throughput(energy_delivered_kwh)
        return energy_delivered_kwh

    def _track_throughput(self, energy_kwh):
        self.total_throughput_mwh += energy_kwh / 1000.0
        self.throughput_since_cycle_max += energy_kwh
        self.operating_hours += energy_kwh / max(self.power_capacity_kw, 1e-6)

    def apply_calendar_aging(self, days=1.0, temp_c=None):
        temp_multiplier = self._temperature_fade_multiplier(temp_c)
        fade_factor = (1.0 - self.calendar_fade_rate * temp_multiplier) ** days
        self.current_capacity_kwh *= fade_factor
        self.capacity_ah = (self.current_capacity_kwh * 1000.0) / self.nominal_voltage
        total_ah = self.available_ah + self.bound_ah
        max_ah = self.capacity_ah
        if total_ah > max_ah:
            scale = max_ah / total_ah
            self.available_ah *= scale
            self.bound_ah *= scale

    def apply_cycle_aging(self, energy_cycled_kwh=None, temp_c=None):
        if energy_cycled_kwh is None:
            energy_cycled_kwh = self.throughput_since_cycle_max
        temp_multiplier = self._temperature_fade_multiplier(temp_c)
        fade = energy_cycled_kwh * self.cycle_fade_per_kwh * temp_multiplier
        self.current_capacity_kwh = max(0.1, self.current_capacity_kwh - fade)
        self.capacity_ah = (self.current_capacity_kwh * 1000.0) / self.nominal_voltage

    def complete_cycle(self):
        soc = self.get_soc()
        if abs(soc - self.soc_at_cycle_start) < 0.05:
            self.cycle_count += 1
            self.throughput_since_cycle_max = 0.0
            self.soc_at_cycle_start = soc

    def is_dead(self):
        if self.current_capacity_kwh <= 0:
            return True
        if self.current_capacity_kwh <= self.nominal_capacity_kwh * self.end_of_life_capacity_fraction:
            return True
        if self.lifetime_throughput_MWh is not None and self.total_throughput_mwh >= self.lifetime_throughput_MWh:
            return True
        if self.lifetime_cycles is not None and self.cycle_count >= self.lifetime_cycles:
            return True
        return False

    def status(self):
        nominal_capacity = max(self.nominal_capacity_kwh, 1e-9)
        return {
            "state_of_charge": self.get_soc(),
            "energy_kwh": self.get_energy_kwh(),
            "capacity_kwh": self.current_capacity_kwh,
            "capacity_fade_percent": (1 - self.current_capacity_kwh / nominal_capacity) * 100.0,
            "max_charge_power_kw": self.get_max_charge_power(),
            "max_discharge_power_kw": self.get_max_discharge_power(),
            "operating_hours": self.operating_hours,
            "total_throughput_mwh": self.total_throughput_mwh,
            "cycle_count": self.cycle_count,
            "is_end_of_life": self.is_dead(),
            "current_temp_c": self.current_temp_c,
        }

    def __repr__(self):
        return (
            f"KiBaMBattery(SOC={self.get_soc()*100:.1f}%, "
            f"Energy={self.get_energy_kwh():.1f} kWh, "
            f"Capacity={self.current_capacity_kwh:.1f} kWh)"
        )
