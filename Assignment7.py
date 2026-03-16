import io
from math import cos, pi, radians, sin

import numpy as np
import pandas as pd

# USER PARAMETERS

LAT = -1.94        # Kigali latitude
TILT = radians(15) # panel tilt
ALBEDO = 0.2

ETA_REF = 0.18
TEMP_COEFF = 0.004
NOCT = 45

SYSTEM_SIZE = 100000  # 100 kWp

# EXTRATERRESTRIAL RADIATION
def extraterrestrial_hour(day_of_year, hour):

    Gsc = 1367

    B = 2*pi*(day_of_year-1)/365

    E0 = 1.00011 + 0.034221*cos(B) + 0.00128*sin(B) \
         + 0.000719*cos(2*B) + 0.000077*sin(2*B)

    # declination (radians)
    decl = radians(23.45*sin(radians(360*(284+day_of_year)/365)))

    # hour angle (radians)
    omega = radians(15*(hour-12))

    phi = radians(LAT)

    # solar altitude
    sin_alt = sin(phi)*sin(decl) + cos(phi)*cos(decl)*cos(omega)

    if sin_alt <= 0:
        return 0.0, 0.0, 0.0

    alt_rad = np.arcsin(sin_alt)

    # solar azimuth (radians) measured from north towards east
    # compute cosine of azimuth
    cos_az = (sin(decl)*cos(phi) - cos(decl)*sin(phi)*cos(omega)) / max(cos(alt_rad), 1e-8)
    # numerical safety
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az = np.arccos(cos_az)
    # determine correct quadrant using hour angle
    if sin(omega) > 0:
        az = 2*pi - az

    G0h = Gsc*E0*sin_alt

    return G0h, alt_rad, az

# TAG MODEL (hourly profile)

def tag_profile(hour):

    sunrise = 6
    sunset = 18

    if hour < sunrise or hour > sunset:
        return 0

    return sin(pi*(hour-sunrise)/(sunset-sunrise))

# ERBS MODEL

def erbs_model(GHI, G0h, sin_alt):

    # GHI and G0h in W/m2
    if G0h <= 0:
        return 0.0, 0.0

    Kt = GHI / G0h if G0h > 0 else 0.0

    if Kt <= 0.22:
        Fd = 1 - 0.09*Kt
    elif Kt <= 0.8:
        Fd = 0.9511 - 0.1604*Kt + 4.388*Kt**2 - 16.638*Kt**3 + 12.336*Kt**4
    else:
        Fd = 0.165

    DHI = Fd*GHI

    # avoid division by zero: sin_alt ~ 0 when sun near horizon
    DNI = (GHI - DHI) / max(sin_alt, 1e-6)

    return DHI, DNI

# TRANSPOSITION MODEL

def transposition(GHI, DHI, DNI, alt_rad, az):
    """Compute irradiance on tilted surface using vector incidence-angle.

    az : solar azimuth (radians) measured from north toward east
    Panel azimuth: facing north => 0 rad. Tilt `TILT` is already in radians.
    """
    # sun direction unit vector (east, north, up)
    cos_alt = max(np.cos(alt_rad), 0.0)
    sx = cos_alt * np.sin(az)
    sy = cos_alt * np.cos(az)
    sz = np.sin(alt_rad)

    # panel normal (east, north, up) — panel azimuth = 0 (north-facing)
    panel_az = 0.0
    nx = np.sin(TILT) * np.sin(panel_az)
    ny = np.sin(TILT) * np.cos(panel_az)
    nz = np.cos(TILT)

    cos_theta = sx*nx + sy*ny + sz*nz
    cos_theta = max(cos_theta, 0.0)

    Gb = DNI * cos_theta
    Gd = DHI * (1 + np.cos(TILT)) / 2
    Gr = GHI * ALBEDO * (1 - np.cos(TILT)) / 2

    return Gb + Gd + Gr


# TEMPERATURE MODEL

def hourly_temperature(Tmin, Tmax, h):
    if 6 <= h <= 18:
        return Tmin + (Tmax - Tmin) * sin(pi * (h - 6) / 12)
    else:
        return Tmin

# CELL TEMPERATURE
# simple NOCT-based model

def cell_temperature(Ta, Gt):
    return Ta + (NOCT - 20) / 800 * Gt


# PV POWER MODEL

def pv_power_efficiency_model(Gt, Tc):
    """Simple efficiency-based PV model (kept for reference).
    Gt in W/m2, Tc in C. Returns power in W.
    """
    eta = ETA_REF * (1 - TEMP_COEFF * (Tc - 25))
    area = SYSTEM_SIZE / (1000 * ETA_REF)
    P = eta * area * Gt
    return max(P, 0.0)


# --- Fill-factor PV model ---
def pv_power_fillfactor(Gt, Tc):
    """Fill-factor based PV model.

    Assumptions (module-level reference values):
    - Module STC power P_stc = 400 W
    - Isc_stc = 10.5 A
    - Voc_stc = 48.0 V
    - FF_stc = 0.78
    - Voc temperature coefficient = -0.0025 / °C (relative)
    - FF temperature degradation ~ -0.0005 / °C

    The model computes Isc proportional to irradiance, Voc adjusted by temperature,
    and P_module = Voc * Isc * FF. The array size SYSTEM_SIZE (W) sets number
    of modules = SYSTEM_SIZE / P_stc.
    """
    # reference module
    P_stc = 400.0
    Isc_stc = 10.5
    Voc_stc = 48.0
    FF_stc = 0.78

    # temperature coefficients
    voc_coeff = -0.0025  # relative per degC
    ff_coeff = -0.0005

    # number of modules to reach SYSTEM_SIZE Wp
    n_modules = max(1, int(round(SYSTEM_SIZE / P_stc)))

    # irradiance ratio (STC=1000 W/m2)
    irr_ratio = max(Gt / 1000.0, 0.0)

    Isc = Isc_stc * irr_ratio
    Voc = Voc_stc * (1 + voc_coeff * (Tc - 25.0))
    FF = FF_stc * (1 + ff_coeff * (Tc - 25.0))

    P_module = Voc * Isc * FF
    P_array = P_module * n_modules
    return max(P_array, 0.0)


def assignment7_pv_power(timestamp, ghi_w_m2, ambient_temp_c, system_size_w=SYSTEM_SIZE):
    """Assignment 7 PV pipeline for one timestamp.

    Starting from horizontal irradiance, this applies:
    1. extraterrestrial radiation / solar position
    2. ERBS diffuse/direct split
    3. tilt transposition
    4. NOCT cell temperature
    5. fill-factor PV power model

    Returns a dict with the main intermediate values so downstream models can
    save traceable PV inputs and outputs.
    """
    day_of_year = pd.Timestamp(timestamp).dayofyear
    hour = pd.Timestamp(timestamp).hour + pd.Timestamp(timestamp).minute / 60.0

    G0h, alt_rad, az = extraterrestrial_hour(day_of_year, hour)
    sin_alt = np.sin(alt_rad) if G0h > 0 else 0.0
    DHI, DNI = erbs_model(ghi_w_m2, G0h, sin_alt)
    tilted_irradiance = transposition(ghi_w_m2, DHI, DNI, alt_rad, az) if G0h > 0 else 0.0
    cell_temp_c = cell_temperature(ambient_temp_c, tilted_irradiance)

    global SYSTEM_SIZE
    previous_system_size = SYSTEM_SIZE
    SYSTEM_SIZE = system_size_w
    try:
        pv_power_w = pv_power_fillfactor(tilted_irradiance, cell_temp_c)
    finally:
        SYSTEM_SIZE = previous_system_size

    module_stc_w = 400.0
    module_count = max(1, int(round(system_size_w / module_stc_w)))

    return {
        "ghi_w_m2": ghi_w_m2,
        "dhi_w_m2": DHI,
        "dni_w_m2": DNI,
        "tilted_irradiance_w_m2": tilted_irradiance,
        "ambient_temp_c": ambient_temp_c,
        "cell_temp_c": cell_temp_c,
        "pv_power_w": pv_power_w,
        "module_count": module_count,
    }

def run_assignment7_from_power_file(
    file_path="POWER_Point_Daily_20250101_20251231_001d94S_030d06E_LST.csv",
    output_path="pv_simulation_output.csv",
):
    """Run the original Assignment 7 workflow from the NASA/POWER daily file."""
    header_row = 0
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("YEAR"):
                header_row = i
                break

    data = pd.read_csv(io.StringIO("".join(lines[header_row:])))
    data["DATE"] = pd.to_datetime(
        data[["YEAR", "MO", "DY"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day"}
        )
    )

    results = []
    for _, row in data.iterrows():
        day = row["DATE"]
        doy = day.dayofyear
        Tmin = row["T2M_MIN"]
        Tmax = row["T2M_MAX"]
        daily_ghi_kwh_per_m2 = row["ALLSKY_SFC_SW_DWN"]

        weights = np.array([tag_profile(h) for h in range(24)])
        if weights.sum() == 0:
            hourly_ghi = np.zeros(24)
        else:
            hourly_ghi = daily_ghi_kwh_per_m2 * 1000.0 * weights / weights.sum()

        for h in range(24):
            ghi = hourly_ghi[h]
            G0h, alt_rad, az = extraterrestrial_hour(doy, h)
            DHI, DNI = erbs_model(ghi, G0h, np.sin(alt_rad) if alt_rad is not None else 0.0)
            Gt = transposition(ghi, DHI, DNI, alt_rad, az)
            Ta = hourly_temperature(Tmin, Tmax, h)
            Tc = cell_temperature(Ta, Gt)
            P = pv_power_fillfactor(Gt, Tc)
            results.append(
                {
                    "datetime": day + pd.Timedelta(hours=h),
                    "GHI": ghi,
                    "DHI": DHI,
                    "DNI": DNI,
                    "Tilt_Irradiance": Gt,
                    "Temperature": Ta,
                    "Cell_Temp": Tc,
                    "PV_Power_W": P,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = run_assignment7_from_power_file()
    annual_energy = df["PV_Power_W"].sum() / 1000.0
    print("Annual PV Energy (kWh):", annual_energy)
