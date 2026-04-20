import calendar

import numpy as np
import pandas as pd
class SolarResourceSimulator:
    """
    Simulate hourly global horizontal irradiance (GHI) from monthly clearness index.
    """

    DEFAULT_TRANSITION_MATRIX = np.array([
        [0.250, 0.179, 0.107, 0.107, 0.143, 0.071, 0.107, 0.036, 0.000, 0.000],
        [0.133, 0.022, 0.089, 0.111, 0.156, 0.178, 0.111, 0.133, 0.067, 0.000],
        [0.064, 0.048, 0.143, 0.048, 0.175, 0.143, 0.206, 0.095, 0.079, 0.000],
        [0.000, 0.022, 0.078, 0.111, 0.156, 0.156, 0.244, 0.167, 0.044, 0.022],
        [0.016, 0.027, 0.037, 0.069, 0.160, 0.219, 0.230, 0.160, 0.075, 0.005],
        [0.013, 0.025, 0.030, 0.093, 0.144, 0.202, 0.215, 0.219, 0.055, 0.004],
        [0.006, 0.041, 0.035, 0.064, 0.090, 0.180, 0.337, 0.192, 0.049, 0.006],
        [0.012, 0.021, 0.029, 0.035, 0.132, 0.123, 0.184, 0.371, 0.082, 0.012],
        [0.008, 0.016, 0.016, 0.024, 0.071, 0.103, 0.159, 0.270, 0.309, 0.024],
        [0.000, 0.000, 0.000, 0.000, 0.059, 0.000, 0.059, 0.294, 0.412, 0.176],
    ], dtype=float)

    def __init__(self, lat=0.0, lon=0.0, transition_matrix=None, random_seed=None):
        self.lat_deg = float(lat)
        self.lon_deg = float(lon)
        self.lat = np.radians(self.lat_deg)
        self.sc = 1367.0
        self.transition_matrix = np.array(
            transition_matrix if transition_matrix is not None else self.DEFAULT_TRANSITION_MATRIX,
            dtype=float,
        )
        if self.transition_matrix.shape != (10, 10):
            raise ValueError("transition_matrix must have shape (10, 10)")

        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Each transition matrix row must contain positive probability mass")
        self.transition_matrix = self.transition_matrix / row_sums
        self.rng = np.random.default_rng(random_seed)

    def get_solar_declination(self, n):
        """Calculate solar declination for day n of the year."""
        return 0.40928 * np.sin(2 * np.pi * (n - 80) / 365)

    def estimate_monthly_kt(self):
        """
        Estimate a generic monthly clearness-index profile from latitude.

        This is a fallback only. Users should prefer measured resource CSVs or
        site-specific monthly Kt values whenever available.
        """
        latitude_scale = min(abs(self.lat_deg) / 45.0, 1.0)
        hemisphere_shift = 0 if self.lat_deg >= 0 else np.pi
        month_angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)

        base = 0.56 - 0.03 * latitude_scale
        seasonal_amplitude = 0.02 + 0.05 * latitude_scale
        monthly_kt = base + seasonal_amplitude * np.sin(month_angle - np.pi / 3 + hemisphere_shift)
        return np.clip(monthly_kt, 0.35, 0.75)

    def generate_daily_sequence(self, avg_kt, days):
        """Generate daily Kt values for one month from a monthly mean Kt."""
        avg_kt = float(np.clip(avg_kt, 0.05, 0.95))
        days = int(days)
        if days <= 0:
            return []

        kt_sequence = []
        current_state = int(np.clip(np.floor(avg_kt * 10), 0, 9))

        for _ in range(days):
            probs = self.transition_matrix[current_state]
            next_state = int(self.rng.choice(np.arange(10), p=probs))
            lower_bound = next_state / 10.0
            kt_val = lower_bound + self.rng.uniform(0.0, 0.1)
            kt_sequence.append(float(np.clip(kt_val, 0.0, 0.88)))
            current_state = next_state
        return kt_sequence

    def generate_hourly_data(self, kt_daily, n):
        """Generate hourly GHI from daily clearness index using a TAG-style profile."""
        kt_daily = float(np.clip(kt_daily, 0.0, 0.88))
        phi1 = 0.35
        delta = self.get_solar_declination(n)
        epsilon = 1 + 0.033 * np.cos(2 * np.pi * n / 365)
        hourly_irradiance = []
        prev_x = 0.0

        for hour in range(24):
            solar_hour = hour + 0.5
            h_angle = (np.pi / 12) * (solar_hour - 12)
            sin_alpha = (
                np.sin(self.lat) * np.sin(delta)
                + np.cos(self.lat) * np.cos(delta) * np.cos(h_angle)
            )

            if sin_alpha <= 0:
                hourly_irradiance.append(0.0)
                continue

            a_term = 0.14 * np.exp(-20 * (kt_daily - 0.35) ** 2)
            b_term = 3 * (kt_daily - 0.45) ** 2 + 16 * kt_daily
            sigma_kt = a_term * np.exp(b_term * (1 - sin_alpha))

            shock = self.rng.normal(0.0, 1.0)
            x_term = phi1 * prev_x + shock * np.sqrt(1 - phi1**2)
            kt_hourly = float(np.clip(kt_daily + x_term * sigma_kt, 0.0, 0.88))

            extraterrestrial_horizontal = self.sc * epsilon * sin_alpha
            hourly_irradiance.append(max(0.0, kt_hourly * extraterrestrial_horizontal))
            prev_x = x_term

        return hourly_irradiance

    def run_full_year(self, monthly_avgs=None, year=2026):
        """Run the full-year simulation from 12 monthly clearness-index values."""
        if monthly_avgs is None:
            monthly_avgs = self.estimate_monthly_kt()

        monthly_avgs = list(monthly_avgs)
        if len(monthly_avgs) != 12:
            raise ValueError("monthly_avgs must contain exactly 12 monthly Kt values")

        results = []
        day_of_year = 1
        month_lengths = [calendar.monthrange(int(year), month)[1] for month in range(1, 13)]

        for month_index, avg_kt in enumerate(monthly_avgs):
            daily_kts = self.generate_daily_sequence(avg_kt, month_lengths[month_index])
            for kt_daily in daily_kts:
                results.extend(self.generate_hourly_data(kt_daily, day_of_year))
                day_of_year += 1
        return results


if __name__ == "__main__":
    sim = SolarResourceSimulator(lat=0.0, lon=0.0, random_seed=42)
    hourly_data = sim.run_full_year()
    df = pd.DataFrame({"hour": range(len(hourly_data)), "ghi_w_m2": hourly_data})
    print(df.head(24))