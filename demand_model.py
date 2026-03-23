import numpy as np
import pandas as pd


class MicrogridLoad:
    """
    Advanced Generic Microgrid Load Model

    Features:
    - Residential / Commercial / Industrial / Custom
    - Flexible timestep (minutes)
    - Daily, Weekly, Annual seasonality
    - Sub-hour interpolation
    - Correlated stochastic variability (AR(1))
    - Technical & Non-Technical losses
    - Load growth
    - Optional holiday effect
    - Fully vectorized (fast)
    """

    def __init__(self,
                 base_kw,
                 timestep_minutes=60, #by default
                 load_type='residential', #'residential', 'commercial', 'industrial', or 'custom'
                 base_year=2026,
                 daily_profile=None,
                 weekly_profile=None,
                 annual_profile=None,
                 variability_std=0.08,
                 ar_coefficient=0.7,
                 growth_rate=0.02, 
                 technical_loss=0.05,
                 non_technical_loss=0.03,
                 holiday_multiplier=0.9,
                 price_elasticity=0.0,
                 tariff_multiplier=1.0):

        if 1440 % timestep_minutes != 0: 
            raise ValueError("timestep_minutes must divide 1440 evenly.")

        self.base_kw = base_kw
        self.timestep_minutes = timestep_minutes
        self.steps_per_day = int(1440 / timestep_minutes)
        self.load_type = load_type
        self.base_year = base_year

        self.daily_profile = (
            self._get_daily_profile(load_type)
            if daily_profile is None else np.array(daily_profile)
        )

        self.weekly_profile = (
            self._get_weekly_profile(load_type)
            if weekly_profile is None else np.array(weekly_profile)
        )

        self.annual_profile = (
            self._get_annual_profile(load_type)
            if annual_profile is None else np.array(annual_profile)
        )

        self.variability_std = variability_std
        self.ar_coefficient = ar_coefficient
        self.growth_rate = growth_rate
        self.technical_loss = technical_loss
        self.non_technical_loss = non_technical_loss
        self.holiday_multiplier = holiday_multiplier
        self.price_elasticity = price_elasticity
        self.tariff_multiplier = max(tariff_multiplier, 1e-6)

    # ---------------------------------------------------------
    # Preset Profiles
    # ---------------------------------------------------------

    @staticmethod
    def _get_daily_profile(load_type):
        profiles = {
            'residential': np.array([
                0.4, 0.35, 0.3, 0.25, 0.3, 0.5, # Early morning ramp
                0.8, 0.9, 0.75, 0.6, 0.55, 0.6, # Midday dip
                0.7, 0.65, 0.6, 0.55, 0.5, 0.65, # Evening peak
                1.0, 1.1, 0.95, 0.85, 0.7, 0.5   # Nighttime drop
            ]),
            'commercial': np.array([
                0.3, 0.25, 0.2, 0.2, 0.25, 0.3,
                0.4, 0.6, 0.85, 0.95, 1.0, 0.95,
                0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
                0.6, 0.65, 0.7, 0.6, 0.45, 0.35
            ]),
            'industrial': np.array([
                0.6, 0.6, 0.6, 0.6, 0.65, 0.75,
                1.0, 1.0, 1.0, 1.0, 0.95, 0.9,
                0.9, 0.9, 0.9, 0.9, 0.85, 0.8,
                0.8, 0.75, 0.75, 0.7, 0.65, 0.6
            ]),
        }
        return profiles.get(load_type, profiles['residential'])

    @staticmethod
    def _get_weekly_profile(load_type):
        profiles = {
            'residential': np.array([1, 1, 1, 1, 1, 1.15, 1.2]),
            'commercial': np.array([1.1, 1.1, 1.1, 1.1, 1.0, 0.6, 0.3]),
            'industrial': np.array([1, 1, 1, 1, 1, 0.7, 0.6]),
        }
        return profiles.get(load_type, profiles['residential'])

    @staticmethod
    def _get_annual_profile(load_type):
        return np.array([
            1.1, 1.05, 0.95, 0.9,
            0.85, 0.9, 1.1, 1.15,
            1.0, 0.95, 1.05, 1.1
        ])

    # ---------------------------------------------------------
    # Time Index
    # ---------------------------------------------------------

    def generate_time_index(self, year=2026, num_days=365):
        freq = f"{self.timestep_minutes}min"
        start = pd.Timestamp(f"{year}-01-01")
        end = start + pd.Timedelta(days=num_days) - pd.Timedelta(minutes=self.timestep_minutes) # Adjust end to be inclusive of the last timestep
        return pd.date_range(start=start, end=end, freq=freq)

    # ---------------------------------------------------------
    # Main Load Generation
    # ---------------------------------------------------------

    def generate_load(self, year=2026, num_days=365):

        time_index = self.generate_time_index(year, num_days)

        # -------- Daily Interpolation --------
        hour_fraction = (
            time_index.hour +
            time_index.minute / 60
        )

        daily_factor = np.interp(
            hour_fraction,
            np.arange(24),
            self.daily_profile
        )

        weekly_factor = self.weekly_profile[time_index.weekday]
        annual_factor = self.annual_profile[time_index.month - 1]

        base_load = (
            self.base_kw *
            daily_factor *
            weekly_factor *
            annual_factor
        )

        # -------- Stochastic Variability --------
        epsilon = np.zeros(len(base_load))
        shocks = np.random.normal(0, self.variability_std, len(base_load))

        for t in range(1, len(base_load)):
            epsilon[t] = (
                self.ar_coefficient * epsilon[t - 1] +
                shocks[t]
            )

        stochastic_factor = np.exp(epsilon)
        load = base_load * stochastic_factor

        # -------- Growth --------
        years_since_base = year - self.base_year
        load *= (1 + self.growth_rate) ** years_since_base

        # -------- Price Elasticity --------
        # A tariff multiplier above 1.0 reduces demand when elasticity is negative.
        load *= self.tariff_multiplier ** self.price_elasticity

        # -------- Loss Modeling --------
        total_loss_fraction = (
            self.technical_loss +
            self.non_technical_loss
        )
        load *= (1 + total_loss_fraction)

        # -------- Simple Holiday Effect --------
        holidays = pd.to_datetime([
            f"{year}-01-01",
            f"{year}-12-25"
        ])

        holiday_mask = time_index.normalize().isin(holidays)
        load[holiday_mask] *= self.holiday_multiplier

        return pd.Series(load, index=time_index, name="Load_kW")

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def generate_load_at_timestamp(self, timestamp):
        """
        Generate load for a single timestamp.
        
        Args:
            timestamp (pd.Timestamp): Specific timestamp to generate load for
        
        Returns:
            float: Load value (kW) for that timestamp
        """
        
        # Daily factor based on hour and minute
        hour_fraction = timestamp.hour + timestamp.minute / 60
        daily_factor = np.interp(
            hour_fraction,
            np.arange(24),
            self.daily_profile
        )
        
        # Weekly factor - convert to int for numpy indexing
        weekly_factor = self.weekly_profile[int(timestamp.weekday())]
        
        # Annual factor - convert to int for numpy indexing
        annual_factor = self.annual_profile[int(timestamp.month) - 1]
        
        base_load = (
            self.base_kw *
            daily_factor *
            weekly_factor *
            annual_factor
        )
        
        # Add small stochastic variability
        stochastic_factor = 1.0 + np.random.normal(0, self.variability_std * 0.5)
        load = base_load * stochastic_factor
        
        # Apply growth
        years_since_base = timestamp.year - self.base_year
        load *= (1 + self.growth_rate) ** years_since_base

        # Apply price elasticity
        load *= self.tariff_multiplier ** self.price_elasticity

        # Apply losses
        total_loss_fraction = (
            self.technical_loss +
            self.non_technical_loss
        )
        load *= (1 + total_loss_fraction)
        
        # Holiday effect
        holidays = pd.to_datetime([
            f"{timestamp.year}-01-01",
            f"{timestamp.year}-12-25"
        ])
        if timestamp.normalize() in holidays.normalize():
            load *= self.holiday_multiplier
        
        return max(load, 0)  # Ensure non-negative

    def get_summary_statistics(self, load_series):

        return {
            "min_kw": load_series.min(),
            "max_kw": load_series.max(),
            "mean_kw": load_series.mean(),
            "std_kw": load_series.std(),
            "peak_to_average_ratio": load_series.max() / load_series.mean(),
            "annual_energy_kwh": load_series.sum() * (self.timestep_minutes / 60),
        }


# ==========================================================
# Generate 1 Year Simulation
# ==========================================================

if __name__ == "__main__":

    model = MicrogridLoad(
        base_kw=30,
        timestep_minutes=30,
        load_type='residential',
        variability_std=0.08,
        ar_coefficient=0.8,
        technical_loss=0.04,
        non_technical_loss=0.03
    )

    load_data = model.generate_load(year=2026, num_days=365)

    stats = model.get_summary_statistics(load_data)

    print("\nMICROGRID LOAD SUMMARY\n")
    for k, v in stats.items():
        print(f"{k}: {v:,.2f}")

    load_data.to_csv("Simeon's_load_1year.csv")
