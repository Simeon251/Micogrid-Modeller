import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from microgrid_simulation import MicrogridSimulation


st.set_page_config(
    page_title="Microgrid Modeler",
    page_icon="M",
    layout="wide",
)


def build_timeseries_figure(results_df, days_to_plot, steps_per_day, battery_min_soc_percent):
    timesteps_to_plot = min(days_to_plot * steps_per_day, len(results_df))
    plot_data = results_df.iloc[:timesteps_to_plot].copy()
    x_axis = plot_data["timestamp"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("white")

    axes[0].plot(x_axis, plot_data["load_kw"], color="black", linewidth=2, linestyle="--", label="Load")
    axes[0].plot(x_axis, plot_data["solar_generation_kw"], color="#d97706", linewidth=1.8, label="PV")
    axes[0].plot(x_axis, plot_data["wind_generation_kw"], color="#0284c7", linewidth=1.8, label="Wind")
    axes[0].plot(x_axis, plot_data["hydro_generation_kw"], color="#0f766e", linewidth=1.8, label="Hydro")
    axes[0].plot(x_axis, plot_data["diesel_generation_kw"], color="#b91c1c", linewidth=1.8, label="Diesel")
    axes[0].set_title("Power Supply vs Load")
    axes[0].set_ylabel("kW")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4)

    axes[1].fill_between(x_axis, 0, plot_data["battery_soc_after"], color="#16a34a", alpha=0.5)
    axes[1].axhline(
        battery_min_soc_percent,
        color="#b91c1c",
        linestyle="--",
        linewidth=1,
        label=f"Min SOC ({battery_min_soc_percent:.0f}%)",
    )
    axes[1].set_title("Battery State of Charge")
    axes[1].set_ylabel("SOC (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].stackplot(
        x_axis,
        plot_data["solar_generation_kw"],
        plot_data["wind_generation_kw"],
        plot_data["hydro_generation_kw"],
        plot_data["diesel_generation_kw"],
        plot_data["battery_discharge_kw"],
        labels=["PV", "Wind", "Hydro", "Diesel", "Battery"],
        colors=["#f59e0b", "#38bdf8", "#14b8a6", "#ef4444", "#22c55e"],
        alpha=0.85,
    )
    axes[2].plot(x_axis, plot_data["load_kw"], color="black", linewidth=2, linestyle="--", label="Load")
    axes[2].set_title("Generation Mix")
    axes[2].set_ylabel("kW")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", ncol=5)

    axes[3].bar(x_axis, plot_data["load_shedding_kw"], color="#dc2626", alpha=0.8, width=0.03)
    axes[3].set_title("Load Shedding")
    axes[3].set_ylabel("kW")
    axes[3].set_xlabel("Time")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def build_energy_mix_figure(metrics):
    labels = ["PV", "Wind", "Hydro", "Diesel", "Battery"]
    values = [
        metrics["total_solar_generation_kwh"],
        metrics["total_wind_generation_kwh"],
        metrics["total_hydro_generation_kwh"],
        metrics["total_diesel_generation_kwh"],
        metrics["total_battery_discharge_kwh"],
    ]
    colors = ["#f59e0b", "#38bdf8", "#14b8a6", "#ef4444", "#22c55e"]

    filtered = [
        (label, value, color)
        for label, value, color in zip(labels, values, colors)
        if value > 1e-6
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if filtered:
        filtered_labels, filtered_values, filtered_colors = zip(*filtered)
        ax.pie(
            filtered_values,
            labels=filtered_labels,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 0.1 else "",
            startangle=90,
            colors=filtered_colors,
        )
    else:
        ax.text(0.5, 0.5, "No energy contribution", ha="center", va="center")
    ax.set_title("Energy Contribution Mix")
    fig.tight_layout()
    return fig


def metrics_csv(metrics):
    metrics_df = pd.DataFrame([metrics])
    return metrics_df.to_csv(index=False).encode("utf-8")


def cashflow_csv(cashflow_df):
    return cashflow_df.to_csv(index=False).encode("utf-8")


def results_csv(results_df):
    export_df = results_df.copy()
    export_df["timestamp"] = export_df["timestamp"].astype(str)
    return export_df.to_csv(index=False).encode("utf-8")


def dataframe_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def build_interpretation(metrics):
    interpretations = []
    recommendations = []

    load_served = metrics.get("load_served_fraction", 0.0)
    renewable_fraction = metrics.get("renewable_fraction", 0.0)
    unmet_load_fraction = metrics.get("unmet_load_fraction", 0.0)
    fuel_used = metrics.get("total_fuel_liters", 0.0)
    lcoe = metrics.get("lcoe", 0.0)
    avg_soc = metrics.get("average_battery_soc", 0.0)
    diesel_runtime = metrics.get("diesel_runtime_hours", 0.0)
    min_dscr = metrics.get("minimum_dscr")
    battery_replacement_interval = metrics.get("battery_replacement_interval_years", 0.0)
    loss_of_load_probability = metrics.get("loss_of_load_probability", 0.0)

    if load_served >= 0.99:
        interpretations.append(
            "Reliability is very strong because almost all demand is being served."
        )
    elif load_served >= 0.95:
        interpretations.append(
            "Reliability is acceptable, but some demand is still not being met."
        )
        recommendations.append(
            "Reduce unmet demand by increasing firm supply, storage, or by lowering peak demand."
        )
    else:
        interpretations.append(
            "Reliability is weak because a noticeable share of the load is not being served."
        )
        recommendations.append(
            "Prioritize reliability improvements by adding generation capacity, battery storage, or revising the dispatch strategy."
        )

    if renewable_fraction >= 0.8:
        interpretations.append(
            "The system is strongly renewable-led, which helps reduce long-term fuel exposure."
        )
    elif renewable_fraction >= 0.5:
        interpretations.append(
            "The system has a balanced renewable contribution, but diesel or other non-renewables still play a meaningful role."
        )
        recommendations.append(
            "Test additional PV, wind, or hydro capacity to further reduce fuel dependence if budget allows."
        )
    else:
        interpretations.append(
            "The system depends heavily on diesel or other non-renewable support."
        )
        recommendations.append(
            "Investigate whether more renewable capacity or better storage sizing can lower diesel dependence and operating cost."
        )

    if unmet_load_fraction > 0.05 or loss_of_load_probability > 0.05:
        interpretations.append(
            "There is a material reliability risk, shown by unmet load or frequent shortage periods."
        )
        recommendations.append(
            "Check the hours with load shedding in the plots and size the system around those shortage periods."
        )

    if fuel_used > 0 and diesel_runtime > 0:
        interpretations.append(
            f"Diesel is actively supporting the microgrid, with {fuel_used:,.1f} L of fuel use over {diesel_runtime:,.1f} runtime hours."
        )
        recommendations.append(
            "If fuel logistics or emissions are a concern, compare this case against a higher-renewable or larger-battery scenario."
        )
    else:
        interpretations.append(
            "Diesel use is negligible or absent in this simulation, so the system is operating mostly without thermal backup."
        )

    if avg_soc < 30:
        interpretations.append(
            "The battery stays at a low average state of charge, which suggests it may be undersized or frequently depleted."
        )
        recommendations.append(
            "Consider increasing battery capacity, reducing discharge stress, or adding more charging energy from renewables."
        )
    elif avg_soc > 85:
        interpretations.append(
            "The battery remains mostly full, which can indicate unused storage potential or oversized energy capacity."
        )
        recommendations.append(
            "Check whether battery size can be optimized downward or whether more renewable energy could be shifted into useful load periods."
        )
    else:
        interpretations.append(
            "Battery utilization appears moderate, which is usually a healthy operating range for daily cycling."
        )

    if battery_replacement_interval > 0:
        if battery_replacement_interval < 5:
            interpretations.append(
                "The battery replacement interval is short, which may increase lifecycle cost."
            )
            recommendations.append(
                "Review depth of discharge, temperature assumptions, and battery sizing to improve battery life."
            )
        elif battery_replacement_interval >= 10:
            interpretations.append(
                "The battery replacement interval is relatively long, which supports better lifecycle economics."
            )

    if lcoe <= 0.2:
        interpretations.append(
            f"The modeled cost of energy is relatively low at ${lcoe:.3f}/kWh."
        )
    elif lcoe <= 0.4:
        interpretations.append(
            f"The modeled cost of energy is moderate at ${lcoe:.3f}/kWh and should be compared with local tariff or benchmark options."
        )
        recommendations.append(
            "Compare the LCOE against the local tariff, diesel-only cost, or your project target before finalizing the design."
        )
    else:
        interpretations.append(
            f"The modeled cost of energy is high at ${lcoe:.3f}/kWh."
        )
        recommendations.append(
            "Review oversizing, fuel use, and component CAPEX assumptions to identify the main cost drivers."
        )

    if pd.notna(min_dscr):
        if min_dscr >= 1.2:
            interpretations.append(
                "Debt service coverage looks healthy under the modeled cash flows."
            )
        elif min_dscr >= 1.0:
            interpretations.append(
                "Debt service coverage is borderline and may be sensitive to cost or tariff changes."
            )
            recommendations.append(
                "Stress-test the project with lower tariffs, higher fuel prices, or lower renewable output before financing decisions."
            )
        else:
            interpretations.append(
                "Debt service coverage is below 1.0 in at least one year, which signals financeability risk."
            )
            recommendations.append(
                "Improve project cash flow through lower CAPEX, lower debt burden, higher tariff, or better system performance."
            )

    deduped_recommendations = list(dict.fromkeys(recommendations))
    return interpretations, deduped_recommendations


st.title("Microgrid Modeler")
st.caption("Enter datasheet values for the currently implemented microgrid resources, storage, and load assumptions, then run the simulation to get interpretable visuals and downloadable results.")

with st.sidebar:
    with st.expander("Help: How To Fill The Inputs", expanded=False):
        st.markdown(
            """
            **Quick Start**

            1. Set the simulation period and dispatch strategy.
            2. Enter component datasheet values for PV, wind, hydro, diesel, and battery.
            3. Provide a measured load or resource CSV if you have one. If not, the model uses its built-in profiles.
            4. Review the economics assumptions before running the model.

            **Best Practice**

            - Start with the default values to confirm the app runs correctly.
            - Change one section at a time so it is easier to understand what affects the results.
            - Use measured resource and load data whenever possible for more realistic outputs.

            **CSV Tip**

            A resource CSV should usually include `timestamp` and any of these columns:
            `ghi_w_m2`, `wind_speed_ms`, `temperature_c`, `hydro_flow_m3s`, `hydro_head_m`, `load_kw`.
            """
        )

    st.header("Simulation Setup")

    timestep_minutes = st.selectbox("Timestep (minutes)", [15, 30, 60], index=2, help="Length of each simulation step. Smaller timesteps capture faster system dynamics but take more computation.")
    num_days = st.slider("Simulation duration (days)", min_value=1, max_value=365, value=30, help="How many days the simulator will model before annualizing the results for lifecycle economics.")
    start_date = st.date_input("Start date", value=pd.Timestamp("2026-01-01"), help="Calendar date used to build the simulation time index.")
    dispatch_strategy = st.selectbox("Dispatch strategy", ["economic_dispatch", "load_following", "cycle_charging"], help="`economic_dispatch` chooses the lowest-cost feasible diesel/battery combination each timestep. `load_following` uses diesel only when needed. `cycle_charging` runs diesel at rated output once started and uses surplus to charge the battery.")
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1, help="Keeps synthetic weather and load generation repeatable so you can compare scenarios fairly.")
    location_lat = st.number_input("Latitude", value=-1.94, format="%.4f", help="Used for synthetic solar, wind, and temperature patterns when measured resource CSV data is not provided.")
    location_lon = st.number_input("Longitude", value=30.06, format="%.4f", help="Used for synthetic resource generation and stored with the scenario assumptions.")
    resource_profile_file = st.text_input("Resource profile CSV path", value="", help="Optional path to a timestamped CSV containing measured or forecast resource data such as GHI, wind speed, temperature, hydro flow, and possibly load.")
    load_profile_file = st.text_input("Load profile CSV path", value="", help="Optional path to a CSV with a measured load profile. Use this when you want to replace the synthetic demand model.")
    st.info(
        "Use `economic_dispatch` to minimize timestep operating cost, `load_following` for renewable-first dispatch, and `cycle_charging` when diesel should charge the battery whenever it starts."
    )
    st.caption("Tip: Measured CSV data usually gives better results than the synthetic defaults.")

    st.header("PV Datasheet")
    pv_capacity_kwp = st.number_input("Array capacity (kWp)", min_value=0.0, value=100.0, step=10.0, help="Installed DC PV array size at standard test conditions.")
    pv_temp_coeff = st.number_input("Temperature coefficient (1/C)", value=-0.00408, format="%.5f", help="Relative power loss per degree Celsius increase in cell temperature above the reference condition.")
    pv_noct = st.number_input("NOCT (C)", min_value=0.0, value=46.0, step=1.0, help="Nominal Operating Cell Temperature from the PV module datasheet.")
    pv_isc_temp_coeff = st.number_input("Isc temperature coefficient (1/C)", value=0.0005, format="%.5f", help="Relative short-circuit current change per degree Celsius.")
    pv_losses = st.slider("System losses", min_value=0.0, max_value=0.5, value=0.15, step=0.01, help="Aggregate non-module PV losses such as wiring, mismatch, soiling, and DC collection losses.")
    pv_inverter_eff = st.slider("Inverter efficiency", min_value=0.5, max_value=1.0, value=0.96, step=0.01, help="Fraction of DC PV power converted to AC output.")
    st.caption("Enter the installed PV size and the main module/system performance assumptions from the datasheet.")

    st.header("Wind Turbine Datasheet")
    wind_capacity_kw = st.number_input("Rated power (kW)", min_value=0.0, value=0.0, step=10.0, help="Nameplate AC output of the wind turbine.")
    wind_swept_area = st.number_input("Swept area (m^2)", min_value=1.0, value=397.6, step=1.0, help="Rotor swept area, usually available from the turbine datasheet.")
    wind_hub_height = st.number_input("Hub height (m)", min_value=1.0, value=34.0, step=1.0, help="Height of the rotor hub above ground. This affects wind-speed extrapolation.")
    wind_cut_in = st.number_input("Cut-in speed (m/s)", min_value=0.0, value=3.5, step=0.1, help="Minimum wind speed at which the turbine starts producing useful power.")
    wind_rated_speed = st.number_input("Rated speed (m/s)", min_value=0.1, value=10.5, step=0.1, help="Wind speed at which the turbine reaches rated output.")
    wind_cut_out = st.number_input("Cut-out speed (m/s)", min_value=0.1, value=20.0, step=0.1, help="Wind speed above which the turbine shuts down for protection.")
    st.caption("Use the turbine datasheet or power curve for these values, especially the cut-in, rated, and cut-out speeds.")

    st.header("Hydropower Datasheet")
    hydro_capacity_kw = st.number_input("Hydro rated power (kW)", min_value=0.0, value=0.0, step=10.0, help="Nameplate output of the hydropower plant or turbine.")
    hydro_design_flow_m3s = st.number_input("Design flow (m^3/s)", min_value=0.0, value=1.0, step=0.1, help="Flow level the hydro turbine is designed to use at rated conditions.")
    hydro_head_m = st.number_input("Net head (m)", min_value=0.0, value=20.0, step=1.0, help="Effective head at the turbine after hydraulic losses.")
    hydro_efficiency = st.slider("Hydro efficiency", min_value=0.1, max_value=1.0, value=0.85, step=0.01, help="Overall turbine-generator conversion efficiency.")
    hydro_min_flow_fraction = st.slider("Hydro minimum flow fraction", min_value=0.0, max_value=1.0, value=0.20, step=0.01, help="Minimum fraction of design flow needed before the hydro unit can operate.")
    hydro_environmental_flow_m3s = st.number_input("Environmental flow (m^3/s)", min_value=0.0, value=0.0, step=0.1, help="Flow that must remain in the river and cannot be diverted to generation.")
    st.caption("If you have a measured river-flow dataset, use it through the resource CSV path for a much more realistic hydro simulation.")

    st.header("Diesel Generator Datasheet")
    diesel_use_kva = st.toggle("Input diesel in kVA", value=True)
    if diesel_use_kva:
        diesel_capacity_kva = st.number_input("Generator rating (kVA)", min_value=0.0, value=60.0, step=5.0, help="Apparent power rating from the generator datasheet.")
        diesel_capacity_kw = 0.0
    else:
        diesel_capacity_kw = st.number_input("Prime rating (kW)", min_value=0.0, value=48.0, step=5.0, help="Real power rating if your datasheet already gives the generator in kW.")
        diesel_capacity_kva = None
    diesel_pf = st.slider("Power factor", min_value=0.5, max_value=1.0, value=0.8, step=0.01, help="Used to convert kVA into kW when the generator is specified in apparent power.")
    diesel_min_load = st.slider("Minimum load fraction", min_value=0.0, max_value=1.0, value=0.25, step=0.01, help="Minimum stable loading fraction. Below this, real engines often operate poorly or should not run continuously.")
    fuel_25 = st.number_input("Fuel use at 25% load (L/h)", min_value=0.0, value=4.5, step=0.1, help="Fuel use from the generator datasheet at 25% loading.")
    fuel_50 = st.number_input("Fuel use at 50% load (L/h)", min_value=0.0, value=7.4, step=0.1, help="Fuel use from the generator datasheet at 50% loading.")
    fuel_75 = st.number_input("Fuel use at 75% load (L/h)", min_value=0.0, value=11.0, step=0.1, help="Fuel use from the generator datasheet at 75% loading.")
    fuel_100 = st.number_input("Fuel use at 100% load (L/h)", min_value=0.0, value=14.7, step=0.1, help="Fuel use from the generator datasheet at full load.")
    st.caption("Fuel values should come from the generator fuel-consumption table at different loading points.")

    st.caption("Diesel Reliability")
    enable_generator_reliability = st.toggle("Enable diesel reliability modeling", value=False, help="Adds planned maintenance and stochastic forced outages using MTBF and MTTR assumptions.")
    mtbf_hours = st.number_input("Diesel MTBF (hours)", min_value=1.0, value=500.0, step=50.0, help="Average operating time between forced failures.")
    mttr_hours = st.number_input("Diesel MTTR (hours)", min_value=1.0, value=8.0, step=1.0, help="Average repair duration after a forced outage.")
    planned_maintenance_interval_hours = st.number_input("Planned maintenance interval (runtime-hours)", min_value=1.0, value=1000.0, step=100.0, help="Runtime interval between planned maintenance shutdowns.")
    planned_maintenance_duration_hours = st.number_input("Planned maintenance duration (hours)", min_value=1.0, value=6.0, step=1.0, help="Length of each planned maintenance event.")

    st.header("Battery Datasheet")
    battery_capacity_kwh = st.number_input("Energy capacity (kWh)", min_value=0.0, value=200.0, step=10.0, help="Total usable battery energy capacity.")
    battery_power_kw = st.number_input("Power capacity (kW)", min_value=0.0, value=60.0, step=5.0, help="Maximum battery charge or discharge power.")
    battery_voltage = st.number_input("Nominal voltage (V)", min_value=1.0, value=48.0, step=1.0, help="Nominal DC battery voltage used in the KiBaM formulation.")
    battery_charge_eff = st.slider("Charge efficiency", min_value=0.5, max_value=1.0, value=0.93, step=0.01, help="Fraction of charging energy that is stored in the battery.")
    battery_discharge_eff = st.slider("Discharge efficiency", min_value=0.5, max_value=1.0, value=0.93, step=0.01, help="Fraction of stored energy that can be delivered back to the system.")
    battery_dod = st.slider("Max depth of discharge", min_value=0.1, max_value=1.0, value=0.80, step=0.01, help="Maximum share of the battery capacity that the controller is allowed to use.")
    battery_k_rate = st.number_input("KiBaM k-rate (1/h)", min_value=0.0, value=0.1, step=0.01, format="%.2f", help="Charge transfer rate between available and bound charge reservoirs in the KiBaM battery model.")
    battery_c_fraction = st.slider("Available charge fraction", min_value=0.05, max_value=0.95, value=0.30, step=0.01, help="Fraction of total battery charge that is immediately available in the KiBaM model.")
    battery_degradation_temp_sensitivity = st.number_input("Battery degradation temp sensitivity", min_value=0.0, value=0.025, step=0.005, format="%.3f", help="Multiplier that increases battery fade above 25 C.")
    battery_eol_fraction = st.slider("Battery end-of-life fraction", min_value=0.50, max_value=0.95, value=0.80, step=0.01, help="Battery replacement threshold as remaining capacity fraction.")
    st.caption("If you are unsure about `k-rate` or `available charge fraction`, keep the defaults unless you have calibration data.")

    st.header("Load Inputs")
    base_load_kw = st.number_input("Base load (kW)", min_value=0.0, value=40.0, step=5.0, help="Average demand anchor for the synthetic load generator.")
    load_type = st.selectbox("Load type", ["residential", "commercial", "industrial"], help="Selects a built-in daily and weekly demand pattern when no measured load CSV is provided.")
    variability_std = st.slider("Load variability", min_value=0.0, max_value=0.3, value=0.08, step=0.01, help="Controls stochastic variability around the base synthetic demand pattern.")
    price_elasticity = st.slider("Load price elasticity", min_value=-1.0, max_value=0.0, value=0.0, step=0.05, help="Negative values reduce load when tariff multipliers rise.")
    tariff_multiplier = st.slider("Tariff multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05, help="Relative retail tariff applied to the synthetic load model.")
    technical_loss = st.slider("Technical loss", min_value=0.0, max_value=0.3, value=0.05, step=0.01, help="Distribution and conversion losses added on top of useful load.")
    non_technical_loss = st.slider("Non-technical loss", min_value=0.0, max_value=0.3, value=0.03, step=0.01, help="Commercial or non-metered losses added on top of useful load.")
    st.caption("These fields matter only when you are using the built-in synthetic load model instead of a measured load CSV.")

    st.caption("Demand-Side Management")
    enable_dsm = st.toggle("Enable DSM", value=False, help="Applies simple load shifting and peak shaving to make demand more flexible.")
    deferrable_load_fraction = st.slider("Deferrable load fraction", min_value=0.0, max_value=0.6, value=0.15, step=0.01, help="Share of peak-period demand that can be shifted to another time window.")
    peak_reduction_fraction = st.slider("Peak reduction fraction", min_value=0.0, max_value=0.5, value=0.05, step=0.01, help="Additional peak-period demand that is curtailed rather than shifted.")
    peak_start_hour = st.slider("Peak start hour", min_value=0, max_value=23, value=18, step=1, help="Start of the demand peak window used by DSM.")
    peak_end_hour = st.slider("Peak end hour", min_value=1, max_value=24, value=22, step=1, help="End of the demand peak window used by DSM.")
    shift_start_hour = st.slider("Shift-to start hour", min_value=0, max_value=23, value=10, step=1, help="Start of the preferred window that receives shifted demand.")
    shift_end_hour = st.slider("Shift-to end hour", min_value=1, max_value=24, value=16, step=1, help="End of the preferred load-shifting window.")

    st.header("Economics")
    project_lifetime_years = st.slider("Project life (years)", min_value=1, max_value=30, value=20, help="Years included in the lifecycle cost and LCOE calculation.")
    nominal_discount_rate = st.slider("Discount rate", min_value=0.0, max_value=0.3, value=0.12, step=0.01, help="Nominal discount rate used to bring future costs and energy to present value.")
    fuel_price_per_liter = st.number_input("Fuel price ($/L)", min_value=0.0, value=1.50, step=0.05, help="Current diesel fuel price used in dispatch and lifecycle economics.")
    fuel_price_escalation_rate = st.slider("Fuel price escalation", min_value=0.0, max_value=0.2, value=0.05, step=0.01, help="Expected annual growth in fuel price over the project lifetime.")
    om_escalation_rate = st.slider("O&M escalation", min_value=0.0, max_value=0.2, value=0.03, step=0.01, help="Expected annual increase in operations and maintenance costs.")
    energy_tariff_per_kwh = st.number_input("Energy tariff ($/kWh)", min_value=0.0, value=0.30, step=0.01, help="Average revenue tariff used for DSCR and project cash flow.")
    tariff_escalation_rate = st.slider("Tariff escalation", min_value=0.0, max_value=0.2, value=0.03, step=0.01, help="Expected annual growth in retail tariff.")
    unserved_energy_cost_per_kwh = st.number_input("Unserved energy penalty ($/kWh)", min_value=0.0, value=2.0, step=0.1, help="Economic penalty assigned to each kWh of unmet demand.")
    st.caption("These values drive lifecycle cost and LCOE, so use assumptions that match your project finance context.")

    st.caption("Capital Cost Assumptions")
    pv_capex_per_kwp = st.number_input("PV CAPEX ($/kWp)", min_value=0.0, value=900.0, step=25.0, help="Installed capital cost per kWp of PV.")
    wind_capex_per_kw = st.number_input("Wind CAPEX ($/kW)", min_value=0.0, value=1500.0, step=50.0, help="Installed capital cost per kW of wind capacity.")
    hydro_capex_per_kw = st.number_input("Hydro CAPEX ($/kW)", min_value=0.0, value=2500.0, step=50.0, help="Installed capital cost per kW of hydropower capacity.")
    diesel_capex_per_kw = st.number_input("Diesel CAPEX ($/kW)", min_value=0.0, value=550.0, step=25.0, help="Installed capital cost per kW of diesel generator capacity.")
    battery_capex_per_kwh = st.number_input("Battery CAPEX ($/kWh)", min_value=0.0, value=350.0, step=10.0, help="Installed battery energy cost per kWh.")
    battery_power_capex_per_kw = st.number_input("Battery power CAPEX ($/kW)", min_value=0.0, value=150.0, step=10.0, help="Additional battery power electronics and converter cost per kW.")

    st.caption("Annual O&M Assumptions")
    pv_fixed_om_per_kw_year = st.number_input("PV fixed O&M ($/kW-yr)", min_value=0.0, value=18.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of PV.")
    wind_fixed_om_per_kw_year = st.number_input("Wind fixed O&M ($/kW-yr)", min_value=0.0, value=45.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of wind.")
    hydro_fixed_om_per_kw_year = st.number_input("Hydro fixed O&M ($/kW-yr)", min_value=0.0, value=35.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of hydropower.")
    diesel_fixed_om_per_kw_year = st.number_input("Diesel fixed O&M ($/kW-yr)", min_value=0.0, value=20.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of diesel capacity.")
    diesel_maintenance_cost_per_hour = st.number_input("Diesel maintenance ($/runtime-hour)", min_value=0.0, value=1.5, step=0.1, help="Maintenance cost applied to each generator runtime hour.")
    battery_fixed_om_per_kwh_year = st.number_input("Battery fixed O&M ($/kWh-yr)", min_value=0.0, value=8.0, step=1.0, help="Annual fixed operations and maintenance cost per kWh of installed battery energy.")
    diesel_variable_om_per_kwh = st.number_input("Diesel variable O&M ($/kWh)", min_value=0.0, value=0.03, step=0.01, format="%.2f", help="Variable cost applied to each kWh produced by the diesel generator.")
    battery_variable_om_per_kwh = st.number_input("Battery variable O&M ($/kWh)", min_value=0.0, value=0.01, step=0.01, format="%.2f", help="Variable cost applied to each kWh discharged from the battery.")
    st.caption("CAPEX is paid up front. Fixed O&M is yearly. Variable O&M scales with energy produced or discharged.")

    st.caption("Debt and Risk")
    debt_fraction = st.slider("Debt fraction", min_value=0.0, max_value=0.95, value=0.70, step=0.05, help="Share of upfront CAPEX financed by debt.")
    debt_interest_rate = st.slider("Debt interest rate", min_value=0.0, max_value=0.25, value=0.10, step=0.01, help="Nominal annual debt interest rate.")
    debt_tenor_years = st.slider("Debt tenor (years)", min_value=1, max_value=20, value=10, help="Years over which the model repays debt.")
    monte_carlo_runs = st.slider("Monte Carlo runs", min_value=0, max_value=1000, value=200, step=50, help="Number of stochastic finance scenarios used for risk outputs.")
    fuel_price_volatility = st.slider("Fuel price volatility", min_value=0.0, max_value=0.6, value=0.18, step=0.01, help="Annualized fuel price volatility used in the GBM process.")
    inflation_volatility = st.slider("Inflation volatility", min_value=0.0, max_value=0.2, value=0.02, step=0.005, help="Shock size for annual CPI in the AR(1) process.")
    exchange_rate_volatility = st.slider("FX volatility", min_value=0.0, max_value=0.4, value=0.08, step=0.01, help="Shock size for annual FX in the AR(1) process.")

    st.header("Visual Range")
    days_to_plot = st.slider("Days to show in time-series plots", min_value=1, max_value=max(1, num_days), value=min(14, num_days), help="How much of the simulated period to show in the time-series charts.")

    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)


if run_simulation:
    with st.spinner("Running microgrid simulation..."):
        sim = MicrogridSimulation(
            timestep_minutes=timestep_minutes,
            num_days=num_days,
            start_date=str(start_date),
            location_lat=location_lat,
            location_lon=location_lon,
            pv_capacity_kwp=pv_capacity_kwp,
            wind_capacity_kw=wind_capacity_kw,
            hydro_capacity_kw=hydro_capacity_kw,
            diesel_capacity_kw=diesel_capacity_kw,
            diesel_capacity_kva=diesel_capacity_kva,
            diesel_power_factor=diesel_pf,
            battery_capacity_kwh=battery_capacity_kwh,
            battery_power_kw=battery_power_kw,
            base_load_kw=base_load_kw,
            load_type=load_type,
            load_profile_file=load_profile_file or None,
            resource_profile_file=resource_profile_file or None,
            dispatch_strategy=dispatch_strategy,
            random_seed=int(random_seed),
            pv_params={
                "temp_coeff_power": pv_temp_coeff,
                "noct": pv_noct,
                "system_losses": pv_losses,
                "inverter_efficiency": pv_inverter_eff,
                "isc_temp_coeff_rel": pv_isc_temp_coeff,
            },
            wind_params={
                "rated_power_kw": wind_capacity_kw,
                "swept_area_m2": wind_swept_area,
                "hub_height_m": wind_hub_height,
                "cut_in": wind_cut_in,
                "rated_speed": wind_rated_speed,
                "cut_out": wind_cut_out,
            },
            hydro_params={
                "rated_power_kw": hydro_capacity_kw,
                "design_flow_m3s": hydro_design_flow_m3s,
                "net_head_m": hydro_head_m,
                "efficiency": hydro_efficiency,
                "min_flow_fraction": hydro_min_flow_fraction,
                "environmental_flow_m3s": hydro_environmental_flow_m3s,
            },
            diesel_params={
                "power_factor": diesel_pf,
                "min_load_factor": diesel_min_load,
                "fuel_curve_lph": {
                    0.25: fuel_25,
                    0.50: fuel_50,
                    0.75: fuel_75,
                    1.0: fuel_100,
                },
                "enable_generator_reliability": enable_generator_reliability,
                "mtbf_hours": mtbf_hours,
                "mttr_hours": mttr_hours,
                "planned_maintenance_interval_hours": planned_maintenance_interval_hours,
                "planned_maintenance_duration_hours": planned_maintenance_duration_hours,
            },
            battery_params={
                "nominal_voltage": battery_voltage,
                "charge_efficiency": battery_charge_eff,
                "discharge_efficiency": battery_discharge_eff,
                "max_depth_of_discharge": battery_dod,
                "k_rate": battery_k_rate,
                "c_fraction": battery_c_fraction,
                "degradation_temp_sensitivity": battery_degradation_temp_sensitivity,
                "end_of_life_capacity_fraction": battery_eol_fraction,
            },
            load_params={
                "variability_std": variability_std,
                "price_elasticity": price_elasticity,
                "tariff_multiplier": tariff_multiplier,
                "technical_loss": technical_loss,
                "non_technical_loss": non_technical_loss,
                "enable_dsm": enable_dsm,
                "deferrable_load_fraction": deferrable_load_fraction,
                "peak_reduction_fraction": peak_reduction_fraction,
                "peak_start_hour": peak_start_hour,
                "peak_end_hour": peak_end_hour,
                "shift_start_hour": shift_start_hour,
                "shift_end_hour": shift_end_hour,
            },
            economic_params={
                "project_lifetime_years": project_lifetime_years,
                "nominal_discount_rate": nominal_discount_rate,
                "fuel_price_per_liter": fuel_price_per_liter,
                "fuel_price_escalation_rate": fuel_price_escalation_rate,
                "om_escalation_rate": om_escalation_rate,
                "energy_tariff_per_kwh": energy_tariff_per_kwh,
                "tariff_escalation_rate": tariff_escalation_rate,
                "unserved_energy_cost_per_kwh": unserved_energy_cost_per_kwh,
                "pv_capex_per_kwp": pv_capex_per_kwp,
                "wind_capex_per_kw": wind_capex_per_kw,
                "hydro_capex_per_kw": hydro_capex_per_kw,
                "diesel_capex_per_kw": diesel_capex_per_kw,
                "battery_capex_per_kwh": battery_capex_per_kwh,
                "battery_power_capex_per_kw": battery_power_capex_per_kw,
                "pv_fixed_om_per_kw_year": pv_fixed_om_per_kw_year,
                "wind_fixed_om_per_kw_year": wind_fixed_om_per_kw_year,
                "hydro_fixed_om_per_kw_year": hydro_fixed_om_per_kw_year,
                "diesel_fixed_om_per_kw_year": diesel_fixed_om_per_kw_year,
                "diesel_maintenance_cost_per_hour": diesel_maintenance_cost_per_hour,
                "battery_fixed_om_per_kwh_year": battery_fixed_om_per_kwh_year,
                "diesel_variable_om_per_kwh": diesel_variable_om_per_kwh,
                "battery_variable_om_per_kwh": battery_variable_om_per_kwh,
                "debt_fraction": debt_fraction,
                "debt_interest_rate": debt_interest_rate,
                "debt_tenor_years": debt_tenor_years,
                "monte_carlo_runs": monte_carlo_runs,
                "fuel_price_volatility": fuel_price_volatility,
                "inflation_volatility": inflation_volatility,
                "exchange_rate_volatility": exchange_rate_volatility,
            },
        )
        results_df = sim.run_simulation(save_results=False, verbose=False)
        metrics = sim.performance_metrics
        cashflow_df = sim.economic_cashflow
        monte_carlo_df = sim.monte_carlo_summary
        monte_carlo_samples_df = sim.monte_carlo_samples
        interpretations, recommendations = build_interpretation(metrics)

    st.success("Simulation complete.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Load Served", f"{metrics['load_served_fraction']:.1%}")
    col2.metric("Renewable Fraction", f"{metrics['renewable_fraction']:.1%}")
    col3.metric("Fuel Used", f"{metrics['total_fuel_liters']:.1f} L")
    col4.metric("LCOE", f"${metrics['lcoe']:.3f}/kWh")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Load", f"{metrics['total_load_kwh']:,.0f} kWh")
    col6.metric("Load Shedding", f"{metrics['total_load_shedding_kwh']:,.1f} kWh")
    col7.metric("Avg Battery SOC", f"{metrics['average_battery_soc']:.1f}%")
    col8.metric("Diesel Runtime", f"{metrics['diesel_runtime_hours']:.1f} h")

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Upfront CAPEX", f"${metrics['upfront_capex']:,.0f}")
    col10.metric("Lifecycle Cost (NPV)", f"${metrics['discounted_lifecycle_cost']:,.0f}")
    col11.metric("Operating Cost/kWh", f"${metrics['operating_cost_per_kwh_served']:.3f}/kWh")
    col12.metric("Battery Replace Interval", f"{metrics['battery_replacement_interval_years']:.1f} yr")

    col13, col14, col15, col16 = st.columns(4)
    col13.metric("Min DSCR", f"{metrics['minimum_dscr']:.2f}" if pd.notna(metrics["minimum_dscr"]) else "N/A")
    col14.metric("Avg DSCR", f"{metrics['average_dscr']:.2f}" if pd.notna(metrics["average_dscr"]) else "N/A")
    col15.metric("Diesel Availability", f"{metrics['diesel_availability_fraction']:.1%}")
    col16.metric("DSM Shifted Energy", f"{metrics['total_dsm_shifted_energy_kwh']:,.1f} kWh")

    col17, col18, col19, col20 = st.columns(4)
    col17.metric("LCOE P50", f"${metrics['lcoe_p50']:.3f}/kWh" if pd.notna(metrics["lcoe_p50"]) else "N/A")
    col18.metric("LCOE P90", f"${metrics['lcoe_p90']:.3f}/kWh" if pd.notna(metrics["lcoe_p90"]) else "N/A")
    col19.metric("Peak Load Before DSM", f"{metrics['peak_baseline_load_kw']:.1f} kW")
    col20.metric("Peak Load After DSM", f"{metrics['peak_load_kw']:.1f} kW")

    st.subheader("Interpretation and Recommendations")
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.markdown("**Interpretation**")
        for item in interpretations:
            st.write(f"- {item}")
    with summary_right:
        st.markdown("**Recommended actions**")
        if recommendations:
            for item in recommendations:
                st.write(f"- {item}")
        else:
            st.write("- The results look broadly balanced. Use scenario comparison to confirm whether this is your preferred design.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visuals", "Results Table", "Economics", "Risk", "Downloads"])

    with tab1:
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(
                build_timeseries_figure(
                    results_df,
                    days_to_plot,
                    sim.steps_per_day,
                    metrics["battery_min_soc_percent"],
                ),
                use_container_width=True,
            )
        with right:
            st.pyplot(build_energy_mix_figure(metrics), use_container_width=True)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Total PV generation (kWh)",
                            "Total wind generation (kWh)",
                            "Total hydro generation (kWh)",
                            "Total diesel generation (kWh)",
                            "Diesel availability",
                            "Forced outage events",
                            "Planned maintenance events",
                            "DSM shifted energy (kWh)",
                            "Peak reduction energy (kWh)",
                            "Direct renewable fraction",
                            "Unmet load fraction",
                            "Loss of load probability",
                            "Peak load (kW)",
                        ],
                        "Value": [
                            round(metrics["total_solar_generation_kwh"], 2),
                            round(metrics["total_wind_generation_kwh"], 2),
                            round(metrics["total_hydro_generation_kwh"], 2),
                            round(metrics["total_diesel_generation_kwh"], 2),
                            round(metrics["diesel_availability_fraction"], 4),
                            int(metrics["diesel_forced_outage_events"]),
                            int(metrics["diesel_planned_outage_events"]),
                            round(metrics["total_dsm_shifted_energy_kwh"], 2),
                            round(metrics["total_peak_reduced_energy_kwh"], 2),
                            round(metrics["direct_renewable_fraction"], 4),
                            round(metrics["unmet_load_fraction"], 4),
                            round(metrics["loss_of_load_probability"], 4),
                            round(metrics["peak_load_kw"], 2),
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        st.dataframe(results_df, use_container_width=True)

    with tab3:
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Upfront CAPEX",
                            "Lifecycle cost (NPV)",
                            "LCOE",
                            "Renewable fraction",
                            "Direct renewable fraction",
                            "Renewable via battery (kWh)",
                            "Diesel availability fraction",
                            "Diesel unavailable hours",
                            "Forced outage events",
                            "Planned maintenance events",
                            "Baseline load energy (kWh)",
                            "DSM shifted energy (kWh)",
                            "Peak reduction energy (kWh)",
                            "Operating cost per kWh",
                            "Discounted operating cost",
                            "Discounted unserved energy cost",
                            "Discounted salvage value",
                            "Annual fixed O&M (base year)",
                            "Annual debt service",
                            "Minimum DSCR",
                            "Average DSCR",
                        ],
                        "Value": [
                            round(metrics["upfront_capex"], 2),
                            round(metrics["discounted_lifecycle_cost"], 2),
                            round(metrics["lcoe"], 4),
                            round(metrics["renewable_fraction"], 4),
                            round(metrics["direct_renewable_fraction"], 4),
                            round(metrics["total_renewable_from_battery_served_kwh"], 2),
                            round(metrics["diesel_availability_fraction"], 4),
                            round(metrics["diesel_unavailable_hours"], 2),
                            int(metrics["diesel_forced_outage_events"]),
                            int(metrics["diesel_planned_outage_events"]),
                            round(metrics["total_baseline_load_kwh"], 2),
                            round(metrics["total_dsm_shifted_energy_kwh"], 2),
                            round(metrics["total_peak_reduced_energy_kwh"], 2),
                            round(metrics["operating_cost_per_kwh_served"], 4),
                            round(metrics["discounted_operating_cost"], 2),
                            round(metrics["discounted_unserved_energy_cost"], 2),
                            round(metrics["discounted_salvage_value"], 2),
                            round(metrics["annual_fixed_om_base"], 2),
                            round(metrics["annual_debt_service"], 2),
                            round(metrics["minimum_dscr"], 3) if pd.notna(metrics["minimum_dscr"]) else None,
                            round(metrics["average_dscr"], 3) if pd.notna(metrics["average_dscr"]) else None,
                        ],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.dataframe(cashflow_df, use_container_width=True, hide_index=True)

    with tab4:
        st.dataframe(monte_carlo_df, use_container_width=True, hide_index=True)
        if not monte_carlo_samples_df.empty:
            st.dataframe(monte_carlo_samples_df, use_container_width=True, hide_index=True)

    with tab5:
        st.download_button(
            "Download results CSV",
            data=results_csv(results_df),
            file_name="microgrid_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download metrics CSV",
            data=metrics_csv(metrics),
            file_name="microgrid_metrics.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download lifecycle cash flow CSV",
            data=cashflow_csv(cashflow_df),
            file_name="microgrid_cashflow.csv",
            mime="text/csv",
        )
        if not monte_carlo_df.empty:
            st.download_button(
                "Download Monte Carlo summary CSV",
                data=dataframe_csv(monte_carlo_df),
                file_name="microgrid_monte_carlo_summary.csv",
                mime="text/csv",
            )
        if not monte_carlo_samples_df.empty:
            st.download_button(
                "Download Monte Carlo samples CSV",
                data=dataframe_csv(monte_carlo_samples_df),
                file_name="microgrid_monte_carlo_samples.csv",
                mime="text/csv",
            )

else:
    st.info("Set the component datasheet values in the sidebar, then click Run Simulation.")

    st.markdown(
        """
        ### What this app does
        - Accepts user-entered datasheet values for the currently implemented microgrid components.
        - Can use timestamped resource CSVs for solar irradiance, wind, temperature, hydro flow, and even load.
        - Runs the existing microgrid simulation engine behind the scenes.
        - Computes lifecycle economics including CAPEX, O&M, fuel-price escalation, replacements, and LCOE.
        - Presents the outputs as summary indicators, clearer charts, and downloadable tables.

        ### Suggested workflow
        1. Enter component ratings from the datasheets and optionally provide resource/load CSV paths.
        2. Adjust simulation duration, dispatch strategy, and economics assumptions.
        3. Run the model and compare reliability, renewable fraction, hydro contribution, fuel use, LCOE, and lifecycle cost.
        """
    )
