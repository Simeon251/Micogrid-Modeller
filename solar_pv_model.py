import io
from math import cos, pi, radians, sin

import numpy as np
import pandas as pd


DEFAULT_LATITUDE_DEG = 0.0
DEFAULT_TILT_DEG = 15.0
DEFAULT_PANEL_AZIMUTH_DEG = 0.0
DEFAULT_ALBEDO = 0.2
DEFAULT_TEMP_COEFF_POWER = -0.00408
DEFAULT_NOCT_C = 46.0
DEFAULT_SYSTEM_LOSSES = 0.15
DEFAULT_INVERTER_EFFICIENCY = 0.96
DEFAULT_SYSTEM_SIZE_W = 100000.0
DEFAULT_MODULE_STC_W = 320.0


def extraterrestrial_hour(day_of_year, hour, lat_deg=DEFAULT_LATITUDE_DEG):
    """Return hourly extraterrestrial horizontal irradiance and sun angles."""
    gsc = 1367.0
    b = 2 * pi * (day_of_year - 1) / 365

    e0 = (
        1.00011
        + 0.034221 * cos(b)
        + 0.00128 * sin(b)
        + 0.000719 * cos(2 * b)
        + 0.000077 * sin(2 * b)
    )

    decl = radians(23.45 * sin(radians(360 * (284 + day_of_year) / 365)))
    omega = radians(15 * (hour - 12))
    phi = radians(lat_deg)

    sin_alt = sin(phi) * sin(decl) + cos(phi) * cos(decl) * cos(omega)
    if sin_alt <= 0:
        return 0.0, 0.0, 0.0

    alt_rad = np.arcsin(sin_alt)
    cos_az = (
        (sin(decl) * cos(phi) - cos(decl) * sin(phi) * cos(omega))
        / max(cos(alt_rad), 1e-8)
    )
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az = np.arccos(cos_az)
    if sin(omega) > 0:
        az = 2 * pi - az

    g0h = gsc * e0 * sin_alt
    return g0h, alt_rad, az


def tag_profile(hour, sunrise=6.0, sunset=18.0):
    if hour < sunrise or hour > sunset:
        return 0.0
    return sin(pi * (hour - sunrise) / max(sunset - sunrise, 1e-6))


def erbs_model(ghi, g0h, sin_alt):
    if g0h <= 0:
        return 0.0, 0.0

    kt = ghi / g0h if g0h > 0 else 0.0
    if kt <= 0.22:
        diffuse_fraction = 1 - 0.09 * kt
    elif kt <= 0.8:
        diffuse_fraction = (
            0.9511
            - 0.1604 * kt
            + 4.388 * kt**2
            - 16.638 * kt**3
            + 12.336 * kt**4
        )
    else:
        diffuse_fraction = 0.165

    diffuse_fraction = float(np.clip(diffuse_fraction, 0.0, 1.0))
    dhi = diffuse_fraction * ghi
    dni = max(0.0, (ghi - dhi) / max(sin_alt, 1e-6))
    return dhi, dni


def transposition(
    ghi,
    dhi,
    dni,
    alt_rad,
    az,
    tilt_deg=DEFAULT_TILT_DEG,
    panel_azimuth_deg=DEFAULT_PANEL_AZIMUTH_DEG,
    albedo=DEFAULT_ALBEDO,
):
    """Transpose irradiance from horizontal plane to the array plane."""
    tilt_rad = radians(tilt_deg)
    panel_azimuth_rad = radians(panel_azimuth_deg)

    cos_alt = max(np.cos(alt_rad), 0.0)
    sx = cos_alt * np.sin(az)
    sy = cos_alt * np.cos(az)
    sz = np.sin(alt_rad)

    nx = np.sin(tilt_rad) * np.sin(panel_azimuth_rad)
    ny = np.sin(tilt_rad) * np.cos(panel_azimuth_rad)
    nz = np.cos(tilt_rad)

    cos_theta = max(0.0, sx * nx + sy * ny + sz * nz)

    beam = dni * cos_theta
    sky_diffuse = dhi * (1 + np.cos(tilt_rad)) / 2
    ground_reflected = ghi * albedo * (1 - np.cos(tilt_rad)) / 2
    return max(0.0, beam + sky_diffuse + ground_reflected)


def cell_temperature(ambient_temp_c, plane_of_array_irradiance_w_m2, noct_c=DEFAULT_NOCT_C):
    return ambient_temp_c + (noct_c - 20.0) / 800.0 * plane_of_array_irradiance_w_m2


def pv_power_efficiency_model(
    plane_of_array_irradiance_w_m2,
    cell_temp_c,
    system_size_w=DEFAULT_SYSTEM_SIZE_W,
    temp_coeff_power=DEFAULT_TEMP_COEFF_POWER,
    system_losses=DEFAULT_SYSTEM_LOSSES,
    inverter_efficiency=DEFAULT_INVERTER_EFFICIENCY,
):
    """Simple AC PV power model using datasheet-style performance adjustments."""
    if plane_of_array_irradiance_w_m2 <= 0 or system_size_w <= 0:
        return 0.0

    temp_factor = 1.0 + temp_coeff_power * (cell_temp_c - 25.0)
    temp_factor = max(temp_factor, 0.0)
    dc_power_w = system_size_w * (plane_of_array_irradiance_w_m2 / 1000.0) * temp_factor
    ac_power_w = dc_power_w * (1.0 - system_losses) * inverter_efficiency
    return max(0.0, ac_power_w)


def pv_power_from_ghi(
    timestamp,
    ghi_w_m2,
    ambient_temp_c,
    system_size_w=DEFAULT_SYSTEM_SIZE_W,
    latitude_deg=DEFAULT_LATITUDE_DEG,
    tilt_deg=DEFAULT_TILT_DEG,
    panel_azimuth_deg=DEFAULT_PANEL_AZIMUTH_DEG,
    albedo=DEFAULT_ALBEDO,
    temp_coeff_power=DEFAULT_TEMP_COEFF_POWER,
    noct_c=DEFAULT_NOCT_C,
    system_losses=DEFAULT_SYSTEM_LOSSES,
    inverter_efficiency=DEFAULT_INVERTER_EFFICIENCY,
):
    """Compute PV output and traceable intermediates from horizontal irradiance."""
    ts = pd.Timestamp(timestamp)
    day_of_year = ts.dayofyear
    hour = ts.hour + ts.minute / 60.0

    g0h, alt_rad, az = extraterrestrial_hour(day_of_year, hour, lat_deg=latitude_deg)
    sin_alt = np.sin(alt_rad) if g0h > 0 else 0.0
    dhi, dni = erbs_model(ghi_w_m2, g0h, sin_alt)
    tilted_irradiance = (
        transposition(
            ghi_w_m2,
            dhi,
            dni,
            alt_rad,
            az,
            tilt_deg=tilt_deg,
            panel_azimuth_deg=panel_azimuth_deg,
            albedo=albedo,
        )
        if g0h > 0
        else 0.0
    )
    cell_temp_c = cell_temperature(ambient_temp_c, tilted_irradiance, noct_c=noct_c)
    pv_power_w = pv_power_efficiency_model(
        tilted_irradiance,
        cell_temp_c,
        system_size_w=system_size_w,
        temp_coeff_power=temp_coeff_power,
        system_losses=system_losses,
        inverter_efficiency=inverter_efficiency,
    )

    return {
        "ghi_w_m2": float(max(0.0, ghi_w_m2)),
        "dhi_w_m2": float(dhi),
        "dni_w_m2": float(dni),
        "tilted_irradiance_w_m2": float(tilted_irradiance),
        "ambient_temp_c": float(ambient_temp_c),
        "cell_temp_c": float(cell_temp_c),
        "pv_power_w": float(pv_power_w),
        "system_size_w": float(system_size_w),
        "latitude_deg": float(latitude_deg),
        "tilt_deg": float(tilt_deg),
        "panel_azimuth_deg": float(panel_azimuth_deg),
        "albedo": float(albedo),
        "system_losses": float(system_losses),
        "inverter_efficiency": float(inverter_efficiency),
        "temp_coeff_power": float(temp_coeff_power),
        "noct_c": float(noct_c),
        "module_count": int(max(1, np.ceil(system_size_w / DEFAULT_MODULE_STC_W))),
        "module_stc_w": float(DEFAULT_MODULE_STC_W),
    }


def run_pv_simulation_from_power_file(
    file_path="POWER_Point_Daily_20250101_20251231_001d94S_030d06E_LST.csv",
    output_path="pv_simulation_output.csv",
    latitude_deg=DEFAULT_LATITUDE_DEG,
    tilt_deg=DEFAULT_TILT_DEG,
    panel_azimuth_deg=DEFAULT_PANEL_AZIMUTH_DEG,
    albedo=DEFAULT_ALBEDO,
    temp_coeff_power=DEFAULT_TEMP_COEFF_POWER,
    noct_c=DEFAULT_NOCT_C,
    system_losses=DEFAULT_SYSTEM_LOSSES,
    inverter_efficiency=DEFAULT_INVERTER_EFFICIENCY,
    system_size_w=DEFAULT_SYSTEM_SIZE_W,
):
    """Run the PV workflow from a NASA/POWER daily file."""
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
        daily_ghi_kwh_per_m2 = row["ALLSKY_SFC_SW_DWN"]
        tmin = row["T2M_MIN"]
        tmax = row["T2M_MAX"]

        weights = np.array([tag_profile(h) for h in range(24)], dtype=float)
        if weights.sum() == 0:
            hourly_ghi = np.zeros(24)
        else:
            hourly_ghi = daily_ghi_kwh_per_m2 * 1000.0 * weights / weights.sum()

        for hour in range(24):
            ambient_temp_c = tmin + (tmax - tmin) * tag_profile(hour)
            pv_state = pv_power_from_ghi(
                timestamp=day + pd.Timedelta(hours=hour),
                ghi_w_m2=float(hourly_ghi[hour]),
                ambient_temp_c=float(ambient_temp_c),
                system_size_w=system_size_w,
                latitude_deg=latitude_deg,
                tilt_deg=tilt_deg,
                panel_azimuth_deg=panel_azimuth_deg,
                albedo=albedo,
                temp_coeff_power=temp_coeff_power,
                noct_c=noct_c,
                system_losses=system_losses,
                inverter_efficiency=inverter_efficiency,
            )
            results.append(
                {
                    "datetime": day + pd.Timedelta(hours=hour),
                    "GHI": pv_state["ghi_w_m2"],
                    "DHI": pv_state["dhi_w_m2"],
                    "DNI": pv_state["dni_w_m2"],
                    "Tilt_Irradiance": pv_state["tilted_irradiance_w_m2"],
                    "Temperature": pv_state["ambient_temp_c"],
                    "Cell_Temp": pv_state["cell_temp_c"],
                    "PV_Power_W": pv_state["pv_power_w"],
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = run_pv_simulation_from_power_file()
    annual_energy = df["PV_Power_W"].sum() / 1000.0
    print("Annual PV Energy (kWh):", annual_energy)
