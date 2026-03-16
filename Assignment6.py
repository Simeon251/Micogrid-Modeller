import numpy as np
import pandas as pd

class SolarSimulatorKigali:
    """
    This class simulates hourly solar irradiance for Kigali using a Markov chain model for daily Kt values and the TAG model for hourly distribution."""
    def __init__(self, lat=-1.94, lon=30.06): #longitude in case we need it for future extensions
        self.lat = np.radians(lat)
        self.sc = 1367  # Solar constant (W/m^2)
        
        # Matrix specifically for Kigali range (approx Kt = 0.54) from 13/Aguiar 88
        self.kigali_matrix = np.array([
           [0.250, 0.179, 0.107, 0.107, 0.143, 0.071, 0.107, 0.036, 0.000, 0.000],
           [0.133, 0.022, 0.089, 0.111, 0.156, 0.178, 0.111, 0.133, 0.067, 0.000],
           [0.064, 0.048, 0.143, 0.048, 0.175, 0.143, 0.206, 0.095, 0.079, 0.000],
           [0.000, 0.022, 0.078, 0.111, 0.156, 0.156, 0.244, 0.167, 0.044, 0.022],
           [0.016, 0.027, 0.037, 0.069, 0.160, 0.219, 0.230, 0.160, 0.075, 0.005],
           [0.013, 0.025, 0.030, 0.093, 0.144, 0.202, 0.215, 0.219, 0.055, 0.004],
           [0.006, 0.041, 0.035, 0.064, 0.090, 0.180, 0.337, 0.192, 0.049, 0.006],
           [0.012, 0.021, 0.029, 0.035, 0.132, 0.123, 0.184, 0.371, 0.082, 0.012],
           [0.008, 0.016, 0.016, 0.024, 0.071, 0.103, 0.159, 0.270, 0.309, 0.024],
           [0.000, 0.000, 0.000, 0.000, 0.059, 0.000, 0.059, 0.294, 0.412, 0.176]
        ])

    def get_solar_declination(self, n):
        """Calculate solar declination for day n of the year."""
        return 0.40928 * np.sin(2 * np.pi * (n - 80) / 365)

    def generate_daily_sequence(self, avg_kt, days):
        """Generate a sequence of daily Kt values based on the average Kt and the Markov chain model."""
        kt_sequence = []
        current_state = int(np.floor(avg_kt * 10))
        
        for _ in range(days):
            probs = self.kigali_matrix[current_state]
            probs /= probs.sum()
            next_state = np.random.choice(range(10), p=probs)
            
            # Step 6 from Notes: Interpolate within the Kt bin
            kt_val = (next_state / 10.0) + np.random.uniform(0, 0.1)
            kt_sequence.append(kt_val)
            current_state = next_state
        return kt_sequence

    def generate_hourly_data(self, Kt_daily, n):
        """TAG Model Implementation"""
        phi1 = 0.35
        delta = self.get_solar_declination(n)
        hourly_irradiance = []
        prev_x = 0 
       
        # Eccentricity correction
        epsilon = 1 + 0.033 * np.cos(2 * np.pi * n / 365)

        for hour in range(1, 25):
            # Solar time and hour angle
            h_angle = (np.pi / 12) * (hour - 12)
            
            # Solar altitude sin(alpha)
            sin_alpha = np.sin(self.lat) * np.sin(delta) + \
                        np.cos(self.lat) * np.cos(delta) * np.cos(h_angle)
            
            if sin_alpha <= 0:
                hourly_irradiance.append(0)
            else:
                # 1. TAG model parameters
                A = 0.14 * np.exp(-20 * (Kt_daily - 0.35)**2)
                B = 3 * (Kt_daily - 0.45)**2 + 16 * Kt_daily
                sigma_kt = A * np.exp(B * (1 - sin_alpha))
                
                # 2. ARMA(1,0) Step
                r = np.random.normal(0, 1)
                x = phi1 * prev_x + r * np.sqrt(1 - phi1**2)
                
                # 3. Calculate hourly kt and extraterrestrial radiation
                kt_h = Kt_daily + x * sigma_kt
                kt_h = max(0, min(kt_h, 0.88)) # Clear sky limit from Aguiar 1992
                
                # Get = Set * eps * sin(alpha)
                G_et = self.sc * epsilon * sin_alpha
                
                # Final Ground Irradiance
                hourly_irradiance.append(kt_h * G_et)
                prev_x = x
                
        return hourly_irradiance

    def run_full_year(self, monthly_avgs):
        """Run the full year simulation given monthly average Kt values."""
        results = []
        day_of_year = 1
        month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        for m_idx, avg_kt in enumerate(monthly_avgs):
            daily_kts = self.generate_daily_sequence(avg_kt, month_lengths[m_idx])
            for kt_d in daily_kts:
                day_hours = self.generate_hourly_data(kt_d, day_of_year)
                results.extend(day_hours)
                day_of_year += 1
        return results

if __name__ == "__main__":
    kigali_monthly_kt = [0.545, 0.562, 0.553, 0.56, 0.559, 0.591, 0.581, 0.559, 0.566, 0.545, 0.536, 0.535]
    sim = SolarSimulatorKigali()
    hourly_data = sim.run_full_year(kigali_monthly_kt)
    df = pd.DataFrame({'Hour': range(len(hourly_data)), 'G_h_W_m2': hourly_data})
    print(df.head(24))
