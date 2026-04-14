"""
Lead-Acid Battery Storage Module

Models lead-acid battery systems for hybrid energy systems with:
- State of charge (SOC) tracking
- Charge/discharge rate limits
- Battery efficiency (round-trip and directional)
- Capacity fade due to cycling and calendar aging
- Temperature effects on performance
- Discharge rate effects (Peukert equation)
- Maximum depth of discharge (DoD) enforcement
- Lifetime tracking (throughput or cycle-based)

Based on typical lead-acid parameters (similar to industrial/solar-grade systems).
"""

import numpy as np


class LeadAcidBattery:
    """Lead-acid battery storage system model.
    
    Key Parameters:
    - energy_capacity_kwh: Total usable energy capacity (kWh)
    - power_capacity_kw: Maximum instantaneous power (kW)
    - nominal_voltage: System voltage (V) - for some calculations
    - charge_efficiency: Charge efficiency (0-1), typically 0.92-0.95
    - discharge_efficiency: Discharge efficiency (0-1), typically 0.92-0.95
    - round_trip_efficiency: Alternative to specifying charge/discharge separately
    - max_dod: Maximum depth of discharge (0-1), typical 0.8 for lead-acid
    - min_soc: Minimum allowable SOC (1 - max_dod)
    """
    
    def __init__(self,
                 energy_capacity_kwh=1000.0,
                 power_capacity_kw=200.0,
                 nominal_voltage=480.0,
                 charge_efficiency=0.93,
                 discharge_efficiency=0.93,
                 round_trip_efficiency=None,
                 max_depth_of_discharge=0.80,
                 temperature_coefficient=-0.003,  # -0.3% per °C deviation from 25°C
                 peukert_exponent=1.24,  # Typical for lead-acid
                 calendar_fade_rate=0.00005,  # Capacity fade per day (~1.8%/yr at reference conditions)
                 cycle_fade_per_kwh=0.0000003,  # Capacity fade per kWh cycled
                 lifetime_throughput_MWh=None,  # Total throughput limit (MWh)
                 lifetime_cycles=None,  # Total cycle limit
                 lifetime_years=10.0):
        """Initialize lead-acid battery model.
        
        Args:
            energy_capacity_kwh: Total energy capacity (kWh)
            power_capacity_kw: Maximum charge/discharge power (kW)
            nominal_voltage: System voltage (V)
            charge_efficiency: Efficiency of charging (0-1)
            discharge_efficiency: Efficiency of discharging (0-1)
            round_trip_efficiency: Alternative to charge/discharge (overrides if set)
            max_depth_of_discharge: MaxDoD as fraction (0-1), default 0.8
            temperature_coefficient: Capacity vs temp coefficient (%/°C)
            peukert_exponent: Peukert exponent for discharge rate effects
            calendar_fade_rate: Daily calendar aging rate (fraction/day)
            cycle_fade_per_kwh: Capacity loss per kWh throughput
            lifetime_throughput_MWh: Total throughput limit for cycle counting
            lifetime_cycles: Alternative cycle-based lifetime
            lifetime_years: Calendar lifetime
        """
        self.energy_capacity_kwh = energy_capacity_kwh
        self.power_capacity_kw = power_capacity_kw
        self.nominal_voltage = nominal_voltage
        
        # Handle efficiency specification
        if round_trip_efficiency is not None:
            # Calculate symmetric charge/discharge from round-trip
            self.charge_efficiency = np.sqrt(round_trip_efficiency)
            self.discharge_efficiency = np.sqrt(round_trip_efficiency)
        else:
            self.charge_efficiency = charge_efficiency
            self.discharge_efficiency = discharge_efficiency
        
        self.max_dod = max_depth_of_discharge
        self.min_soc = 1.0 - max_depth_of_discharge
        
        # Temperature and rate effects
        self.reference_temp_c = 25.0
        self.temp_coefficient = temperature_coefficient
        self.peukert_exponent = peukert_exponent
        
        # Aging and degradation
        self.calendar_fade_rate = calendar_fade_rate  # per day
        self.cycle_fade_per_kwh = cycle_fade_per_kwh
        self.nominal_capacity_kwh = energy_capacity_kwh
        self.current_capacity_kwh = energy_capacity_kwh
        self.lifetime_throughput_MWh = lifetime_throughput_MWh
        self.lifetime_cycles = lifetime_cycles
        self.lifetime_years = lifetime_years
        
        # State variables
        self.state_of_charge = 1.0  # Start at 100% SOC
        self.energy_kwh = self.state_of_charge * self.current_capacity_kwh
        self.operating_hours = 0.0
        self.total_throughput_mwh = 0.0
        self.cycle_count = 0.0
        self.throughput_since_cycle_max = 0.0  # Energy cycled in current cycle
        self.soc_at_cycle_start = 1.0
        
        # Temperature tracking
        self.current_temp_c = 25.0
        
    # ================================================================
    # State of Charge and Capacity Management
    # ================================================================
    
    def get_energy_available(self):
        """Return currently available energy (kWh) for discharge."""
        return self.energy_kwh
    
    def get_energy_to_full(self):
        """Return energy needed to charge to 100% SOC (kWh)."""
        return self.current_capacity_kwh - self.energy_kwh
    
    def get_soc(self):
        """Return current state of charge (0-1)."""
        return self.state_of_charge
    
    def get_soc_min(self):
        """Return minimum allowable SOC based on DoD limit."""
        return self.min_soc
    
    def is_dead(self):
        """Check if battery has reached end of life."""
        if self.current_capacity_kwh <= 0:
            return True
        if self.lifetime_throughput_MWh is not None:
            if self.total_throughput_mwh >= self.lifetime_throughput_MWh:
                return True
        if self.lifetime_cycles is not None:
            if self.cycle_count >= self.lifetime_cycles:
                return True
        return False
    
    # ================================================================
    # Charge/Discharge Rate Limits
    # ================================================================
    
    def get_max_charge_power(self, temp_c=None, timestep_hr=0.25):
        """Return maximum charge power (kW) at this timestep.
        
        Accounts for:
        - Physical power limit
        - Temperature derating
        - Current capacity (can't charge beyond full)
        """
        if temp_c is None:
            temp_c = self.current_temp_c
        
        # Temperature derating: capacity drops at extreme temperatures
        temp_factor = 1.0 + self.temp_coefficient * (temp_c - self.reference_temp_c)
        temp_factor = max(0.3, min(1.0, temp_factor))  # Clamp to 30%-100%
        
        max_power = self.power_capacity_kw * temp_factor
        
        # Can't charge beyond capacity: convert available energy to a power limit
        energy_to_full = self.get_energy_to_full()
        if timestep_hr <= 0:
            timestep_hr = 0.25
        max_power = min(max_power, energy_to_full / timestep_hr)
        
        return max(0, max_power)
    
    def get_max_discharge_power(self, temp_c=None, timestep_hr=0.25):
        """Return maximum discharge power (kW) at this timestep.
        
        Accounts for:
        - Physical power limit
        - Temperature derating
        - Available energy and min SOC constraint
        - Peukert effect (discharge rate reduces available capacity)
        """
        if temp_c is None:
            temp_c = self.current_temp_c
        
        # Temperature derating
        temp_factor = 1.0 + self.temp_coefficient * (temp_c - self.reference_temp_c)
        temp_factor = max(0.3, min(1.0, temp_factor))
        
        max_power = self.power_capacity_kw * temp_factor
        
        # Can't discharge below min_soc
        energy_available = max(0, self.energy_kwh - self.min_soc * self.current_capacity_kwh)
        if timestep_hr <= 0:
            timestep_hr = 0.25
        max_power = min(max_power, energy_available / timestep_hr)
        
        # Peukert effect: higher discharge rates reduce effective capacity
        # Effective capacity = nominal * (C_rate / C_nominal)^(1 - k)
        # where k is Peukert exponent (typically 1.2-1.3)
        # This is an approximation - a more detailed model would use Peukert's equation
        # C_rate = power_kw / (capacity_kwh / 1.0)  [for 1-hour reference]
        if max_power > 0:
            c_rate = max_power / self.current_capacity_kwh
            peukert_derating = (c_rate / 1.0) ** (1.0 - self.peukert_exponent)
            peukert_derating = max(0.3, min(1.0, peukert_derating))  # Clamp
            max_power *= peukert_derating
        
        return max(0, max_power)
    
    # ================================================================
    # Charge/Discharge Operations
    # ================================================================
    
    def charge(self, power_kw, timestep_hr=0.25, temp_c=None):
        """Charge battery with given power for duration of timestep.
        
        Args:
            power_kw: Requested charge power (kW)
            timestep_hr: Duration of timestep (hours)
            temp_c: Ambient/cell temperature (°C)
        
        Returns:
            Actual energy charged (kWh), accounting for efficiency and limits
        """
        if temp_c is None:
            temp_c = self.current_temp_c
        self.current_temp_c = temp_c
        
        max_charge = self.get_max_charge_power(temp_c, timestep_hr=timestep_hr)
        actual_power = min(power_kw, max_charge)
        
        # Energy input required (accounting for charge efficiency)
        energy_input = actual_power * timestep_hr / self.charge_efficiency
        actual_energy_stored = actual_power * timestep_hr
        
        # Update state
        self.energy_kwh = min(self.current_capacity_kwh, 
                              self.energy_kwh + actual_energy_stored)
        self.state_of_charge = self.energy_kwh / self.current_capacity_kwh
        
        # Track throughput
        self.total_throughput_mwh += actual_energy_stored / 1000.0
        self.throughput_since_cycle_max += actual_energy_stored
        self.operating_hours += timestep_hr
        
        return actual_energy_stored
    
    def discharge(self, power_kw, timestep_hr=0.25, temp_c=None):
        """Discharge battery with given power for duration of timestep.
        
        Args:
            power_kw: Requested discharge power (kW)
            timestep_hr: Duration of timestep (hours)
            temp_c: Ambient/cell temperature (°C)
        
        Returns:
            Actual energy discharged (kWh), accounting for efficiency and limits
        """
        if temp_c is None:
            temp_c = self.current_temp_c
        self.current_temp_c = temp_c
        
        max_discharge = self.get_max_discharge_power(temp_c, timestep_hr=timestep_hr)
        actual_power = min(power_kw, max_discharge)
        
        # Energy output (accounting for discharge efficiency)
        energy_output = actual_power * timestep_hr
        energy_withdrawn = energy_output / self.discharge_efficiency
        
        # Update state
        self.energy_kwh = max(self.min_soc * self.current_capacity_kwh,
                              self.energy_kwh - energy_withdrawn)
        self.state_of_charge = self.energy_kwh / self.current_capacity_kwh
        
        # Track throughput
        self.total_throughput_mwh += energy_output / 1000.0
        self.throughput_since_cycle_max += energy_output
        self.operating_hours += timestep_hr
        
        return energy_output
    
    # ================================================================
    # Degradation and Aging
    # ================================================================
    
    def apply_calendar_aging(self, days=1.0):
        """Apply calendar-based capacity fade.
        
        Args:
            days: Number of days elapsed
        """
        fade_factor = (1.0 - self.calendar_fade_rate) ** days
        self.current_capacity_kwh *= fade_factor
        
        # Adjust energy to stay within new capacity
        self.energy_kwh = min(self.energy_kwh, self.current_capacity_kwh)
        self.state_of_charge = self.energy_kwh / self.current_capacity_kwh
    
    def apply_cycle_aging(self, energy_cycled_kwh=None):
        """Apply cycle-based capacity fade.
        
        Can be called after each cycle or each timestep with energy throughput.
        
        Args:
            energy_cycled_kwh: Energy cycled since last call (if None, uses throughput)
        """
        if energy_cycled_kwh is None:
            energy_cycled_kwh = self.throughput_since_cycle_max
        
        # Simple linear model: capacity fade proportional to throughput
        fade = energy_cycled_kwh * self.cycle_fade_per_kwh
        self.current_capacity_kwh = max(0.1, self.current_capacity_kwh - fade)
        
        # Adjust energy to stay within new capacity
        self.energy_kwh = min(self.energy_kwh, self.current_capacity_kwh)
        self.state_of_charge = self.energy_kwh / self.current_capacity_kwh
    
    def complete_cycle(self):
        """Mark end of a discharge/charge cycle for tracking.
        
        Increment cycle counter and reset cycle tracking.
        """
        # Simple cycle detection: when SOC returns to near starting point
        if abs(self.state_of_charge - self.soc_at_cycle_start) < 0.05:
            self.cycle_count += 1.0
            self.throughput_since_cycle_max = 0.0
            self.soc_at_cycle_start = self.state_of_charge
    
    def get_capacity_fade_percent(self):
        """Return capacity fade as percentage relative to nominal."""
        if self.nominal_capacity_kwh <= 0:
            return 100.0
        fade_percent = (1.0 - self.current_capacity_kwh / self.nominal_capacity_kwh) * 100.0
        return max(0, min(100, fade_percent))
    
    # ================================================================
    # Status and Reporting
    # ================================================================
    
    def status(self):
        """Return dictionary with comprehensive battery status."""
        return {
            'energy_kwh': self.energy_kwh,
            'state_of_charge': self.state_of_charge,
            'capacity_kwh': self.current_capacity_kwh,
            'capacity_fade_percent': self.get_capacity_fade_percent(),
            'max_charge_power_kw': self.get_max_charge_power(),
            'max_discharge_power_kw': self.get_max_discharge_power(),
            'operating_hours': self.operating_hours,
            'total_throughput_mwh': self.total_throughput_mwh,
            'cycle_count': self.cycle_count,
            'is_end_of_life': self.is_dead(),
            'current_temp_c': self.current_temp_c
        }
    
    def __repr__(self):
        """String representation of battery state."""
        return (f"LeadAcidBattery(SOC={self.state_of_charge*100:.1f}%, "
                f"Energy={self.energy_kwh:.1f} kWh, "
                f"Capacity={self.current_capacity_kwh:.1f} kWh, "
                f"Fade={self.get_capacity_fade_percent():.1f}%)")


class KiBaMBattery:
    """Kinetic Battery Model (KiBaM) based battery storage.

    Implements a two-tank model where charge is distributed between an
    "available" and a "bound" reservoir. State updates are driven by current
    (A) rather than power (kW), which allows for more accurate modeling of
    charge/discharge dynamics and battery kinetics.

    Key references:
    - Manwell and McGowan, "Renewable Energy Systems" (KiBaM description)
    - Typical lead-acid behavior (efficiency, SOC limits, etc.)
    """

    def __init__(self,
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
                 lifetime_years=10.0):
        """Initialize KiBaM battery.

        Args:
            energy_capacity_kwh: Nominal energy storage capacity (kWh)
            power_capacity_kw: Maximum charge/discharge power (kW)
            nominal_voltage: Nominal battery voltage (V)
            charge_efficiency: Charging efficiency (0-1)
            discharge_efficiency: Discharging efficiency (0-1)
            max_depth_of_discharge: Maximum fraction of capacity that can be used
            k_rate: Rate constant (1/h) for charge transfer between reservoirs
            c_fraction: Fraction of capacity in the "available" reservoir
        """
        self.energy_capacity_kwh = energy_capacity_kwh
        self.power_capacity_kw = power_capacity_kw
        self.nominal_voltage = nominal_voltage

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_dod = max_depth_of_discharge
        self.min_soc = 1.0 - max_depth_of_discharge

        self.k_rate = k_rate
        self.c_fraction = c_fraction

        # Convert energy capacity to amp-hours for current-based model
        self.capacity_ah = (energy_capacity_kwh * 1000.0) / nominal_voltage
        self.available_ah = self.c_fraction * self.capacity_ah
        self.bound_ah = (1.0 - self.c_fraction) * self.capacity_ah

        # Aging parameters
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

        # State tracking
        self.state_of_charge = 1.0
        self.total_throughput_mwh = 0.0
        self.cycle_count = 0.0
        self.soc_at_cycle_start = 1.0
        self.throughput_since_cycle_max = 0.0
        self.operating_hours = 0.0
        self.current_temp_c = 25.0

    def _temperature_fade_multiplier(self, temp_c=None):
        """Return a degradation multiplier relative to 25 C."""
        if temp_c is None:
            temp_c = self.current_temp_c
        return max(0.25, 1.0 + self.degradation_temp_sensitivity * (temp_c - 25.0))

    def _current_from_power(self, power_kw):
        """Convert power (kW) to current (A) using nominal voltage."""
        return (power_kw * 1000.0) / max(self.nominal_voltage, 1e-6)

    def get_energy_kwh(self):
        """Return energy currently stored in kWh."""
        return (self.available_ah + self.bound_ah) * self.nominal_voltage / 1000.0

    def get_soc(self):
        """Return state of charge [0-1]."""
        return min(1.0, max(0.0, self.get_energy_kwh() / self.current_capacity_kwh))

    def get_min_soc(self):
        return self.min_soc

    def get_max_charge_power(self, timestep_hr=1.0):
        """Return max charge input power (kW) given current state and timestep."""
        # Physical power limit
        max_input_power = self.power_capacity_kw

        # Capacity limit: cannot store more energy than remaining capacity
        energy_to_full = self.current_capacity_kwh - self.get_energy_kwh()
        if energy_to_full <= 0:
            return 0.0
        # Need to supply more energy than stored due to efficiency losses
        max_input_by_capacity = energy_to_full / max(timestep_hr, 1e-6) / max(self.charge_efficiency, 1e-6)

        return min(max_input_power, max_input_by_capacity)

    def get_max_discharge_power(self, timestep_hr=1.0):
        """Return max discharge output power (kW) given current state and timestep."""
        # Cannot discharge below minimum SOC
        energy_available = self.get_energy_kwh() - self.min_soc * self.current_capacity_kwh
        if energy_available <= 0:
            return 0.0

        # Output energy is reduced by discharge efficiency
        max_output_by_capacity = energy_available * max(self.discharge_efficiency, 1e-6) / max(timestep_hr, 1e-6)
        return min(self.power_capacity_kw, max_output_by_capacity)

    def charge(self, power_kw, timestep_hr=1.0, temp_c=None):
        """Charge the battery using power (kW) for timestep (h).

        Returns:
            energy_drawn_kwh (float): Energy drawn from the system (kWh)
        """
        if power_kw <= 0:
            return 0.0

        if temp_c is not None:
            self.current_temp_c = temp_c

        # Limit input power by battery capability and remaining capacity
        max_input_power = self.get_max_charge_power(timestep_hr=timestep_hr)
        actual_input_power = min(power_kw, max_input_power)

        # Energy drawn from system
        energy_drawn_kwh = actual_input_power * timestep_hr
        # Energy stored after efficiency losses
        energy_stored_kwh = energy_drawn_kwh * self.charge_efficiency

        # Convert stored energy to amp-hours for KiBaM update
        delta_ah = energy_stored_kwh * 1000.0 / max(self.nominal_voltage, 1e-6)

        # Update KiBaM reservoirs
        available_delta = self.c_fraction * delta_ah + self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr
        bound_delta = (1.0 - self.c_fraction) * delta_ah - self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr

        self.available_ah = max(0.0, min(self.capacity_ah, self.available_ah + available_delta))
        self.bound_ah = max(0.0, min(self.capacity_ah - self.available_ah, self.bound_ah + bound_delta))

        self._track_throughput(energy_stored_kwh)
        return energy_drawn_kwh

    def discharge(self, power_kw, timestep_hr=1.0, temp_c=None):
        """Discharge the battery to deliver power (kW) for timestep (h).

        Returns:
            energy_delivered_kwh (float): Energy delivered to load (kWh)
        """
        if power_kw <= 0:
            return 0.0

        if temp_c is not None:
            self.current_temp_c = temp_c

        # Limit output power by capability and available energy
        max_output_power = self.get_max_discharge_power(timestep_hr=timestep_hr)
        actual_output_power = min(power_kw, max_output_power)

        # Energy delivered to load
        energy_delivered_kwh = actual_output_power * timestep_hr
        # Energy removed from battery accounting for efficiency losses
        energy_removed_kwh = energy_delivered_kwh / max(self.discharge_efficiency, 1e-6)

        # Convert to amp-hours for KiBaM update
        delta_ah = energy_removed_kwh * 1000.0 / max(self.nominal_voltage, 1e-6)

        # Apply KiBaM discharge
        available_delta = -self.c_fraction * delta_ah + self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr
        bound_delta = -(1.0 - self.c_fraction) * delta_ah - self.k_rate * (self.bound_ah - self.available_ah) * timestep_hr

        self.available_ah = max(0.0, self.available_ah + available_delta)
        self.bound_ah = max(0.0, self.bound_ah + bound_delta)

        self._track_throughput(energy_delivered_kwh)
        return energy_delivered_kwh

    def _track_throughput(self, energy_kwh):
        """Track energy throughput for aging calculations."""
        self.total_throughput_mwh += energy_kwh / 1000.0
        self.throughput_since_cycle_max += energy_kwh
        self.operating_hours += energy_kwh / max(self.power_capacity_kw, 1e-6)

    def apply_calendar_aging(self, days=1.0, temp_c=None):
        """Apply simple calendar aging."""
        temp_multiplier = self._temperature_fade_multiplier(temp_c)
        fade_factor = (1.0 - self.calendar_fade_rate * temp_multiplier) ** days
        self.current_capacity_kwh *= fade_factor
        # Adjust capacity in Ah to match kWh
        self.capacity_ah = (self.current_capacity_kwh * 1000.0) / self.nominal_voltage
        # Ensure stored energy doesn't exceed capacity
        total_ah = self.available_ah + self.bound_ah
        max_ah = self.capacity_ah
        if total_ah > max_ah:
            scale = max_ah / total_ah
            self.available_ah *= scale
            self.bound_ah *= scale

    def apply_cycle_aging(self, energy_cycled_kwh=None, temp_c=None):
        """Apply cycle aging based on energy throughput."""
        if energy_cycled_kwh is None:
            energy_cycled_kwh = self.throughput_since_cycle_max
        temp_multiplier = self._temperature_fade_multiplier(temp_c)
        fade = energy_cycled_kwh * self.cycle_fade_per_kwh * temp_multiplier
        self.current_capacity_kwh = max(0.1, self.current_capacity_kwh - fade)
        self.capacity_ah = (self.current_capacity_kwh * 1000.0) / self.nominal_voltage

    def complete_cycle(self):
        """Record a full charge/discharge cycle."""
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
        return {
            'state_of_charge': self.get_soc(),
            'energy_kwh': self.get_energy_kwh(),
            'capacity_kwh': self.current_capacity_kwh,
            'capacity_fade_percent': (1 - self.current_capacity_kwh / self.nominal_capacity_kwh) * 100.0,
            'max_charge_power_kw': self.get_max_charge_power(),
            'max_discharge_power_kw': self.get_max_discharge_power(),
            'operating_hours': self.operating_hours,
            'total_throughput_mwh': self.total_throughput_mwh,
            'cycle_count': self.cycle_count,
            'is_end_of_life': self.is_dead(),
            'current_temp_c': self.current_temp_c,
        }

    def __repr__(self):
        return (f"KiBaMBattery(SOC={self.get_soc()*100:.1f}%, "
                f"Energy={self.get_energy_kwh():.1f} kWh, "
                f"Capacity={self.current_capacity_kwh:.1f} kWh)")
