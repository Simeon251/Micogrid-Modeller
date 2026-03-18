import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from microgrid_simulation import MicrogridSimulation


st.set_page_config(
    page_title="Microgrid Modeler",
    page_icon="M",
    layout="wide",
)


def build_timeseries_figure(results_df, days_to_plot, steps_per_day):
    timesteps_to_plot = min(days_to_plot * steps_per_day, len(results_df))
    plot_data = results_df.iloc[:timesteps_to_plot].copy()
    x_axis = plot_data["timestamp"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("white")

    axes[0].plot(x_axis, plot_data["load_kw"], color="black", linewidth=2, linestyle="--", label="Load")
    axes[0].plot(x_axis, plot_data["solar_generation_kw"], color="#d97706", linewidth=1.8, label="PV")
    axes[0].plot(x_axis, plot_data["wind_generation_kw"], color="#0284c7", linewidth=1.8, label="Wind")
    axes[0].plot(x_axis, plot_data["diesel_generation_kw"], color="#b91c1c", linewidth=1.8, label="Diesel")
    axes[0].set_title("Power Supply vs Load")
    axes[0].set_ylabel("kW")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4)

    axes[1].fill_between(x_axis, 0, plot_data["battery_soc_after"], color="#16a34a", alpha=0.5)
    axes[1].axhline(40, color="#b91c1c", linestyle="--", linewidth=1, label="Dispatch Min SOC")
    axes[1].set_title("Battery State of Charge")
    axes[1].set_ylabel("SOC (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].stackplot(
        x_axis,
        plot_data["solar_generation_kw"],
        plot_data["wind_generation_kw"],
        plot_data["diesel_generation_kw"],
        plot_data["battery_discharge_kw"],
        labels=["PV", "Wind", "Diesel", "Battery"],
        colors=["#f59e0b", "#38bdf8", "#ef4444", "#22c55e"],
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
    labels = ["PV", "Wind", "Diesel", "Battery"]
    values = [
        metrics["total_solar_generation_kwh"],
        metrics["total_wind_generation_kwh"],
        metrics["total_diesel_generation_kwh"],
        metrics["total_battery_discharge_kwh"],
    ]
    colors = ["#f59e0b", "#38bdf8", "#ef4444", "#22c55e"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title("Energy Contribution Mix")
    fig.tight_layout()
    return fig


def metrics_csv(metrics):
    metrics_df = pd.DataFrame([metrics])
    return metrics_df.to_csv(index=False).encode("utf-8")


def results_csv(results_df):
    export_df = results_df.copy()
    export_df["timestamp"] = export_df["timestamp"].astype(str)
    return export_df.to_csv(index=False).encode("utf-8")


st.title("Microgrid Modeler")
st.caption("Enter datasheet values for the currently implemented microgrid resources, storage, and load assumptions, then run the simulation to get interpretable visuals and downloadable results.")

with st.sidebar:
    st.header("Simulation Setup")

    timestep_minutes = st.selectbox("Timestep (minutes)", [15, 30, 60], index=2)
    num_days = st.slider("Simulation duration (days)", min_value=1, max_value=365, value=30)
    start_date = st.date_input("Start date", value=pd.Timestamp("2026-01-01"))
    dispatch_strategy = st.selectbox("Dispatch strategy", ["load_following", "cycle_charging"])
    random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("PV Datasheet")
    pv_capacity_kwp = st.number_input("Array capacity (kWp)", min_value=0.0, value=100.0, step=10.0)
    pv_temp_coeff = st.number_input("Temperature coefficient (1/C)", value=-0.00408, format="%.5f")
    pv_noct = st.number_input("NOCT (C)", min_value=0.0, value=46.0, step=1.0)
    pv_losses = st.slider("System losses", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
    pv_inverter_eff = st.slider("Inverter efficiency", min_value=0.5, max_value=1.0, value=0.96, step=0.01)

    st.header("Wind Turbine Datasheet")
    wind_capacity_kw = st.number_input("Rated power (kW)", min_value=0.0, value=0.0, step=10.0)
    wind_swept_area = st.number_input("Swept area (m^2)", min_value=1.0, value=397.6, step=1.0)
    wind_hub_height = st.number_input("Hub height (m)", min_value=1.0, value=34.0, step=1.0)
    wind_cut_in = st.number_input("Cut-in speed (m/s)", min_value=0.0, value=3.5, step=0.1)
    wind_rated_speed = st.number_input("Rated speed (m/s)", min_value=0.1, value=10.5, step=0.1)
    wind_cut_out = st.number_input("Cut-out speed (m/s)", min_value=0.1, value=20.0, step=0.1)

    st.header("Diesel Generator Datasheet")
    diesel_use_kva = st.toggle("Input diesel in kVA", value=True)
    if diesel_use_kva:
        diesel_capacity_kva = st.number_input("Generator rating (kVA)", min_value=0.0, value=60.0, step=5.0)
        diesel_capacity_kw = 0.0
    else:
        diesel_capacity_kw = st.number_input("Prime rating (kW)", min_value=0.0, value=48.0, step=5.0)
        diesel_capacity_kva = None
    diesel_pf = st.slider("Power factor", min_value=0.5, max_value=1.0, value=0.8, step=0.01)
    diesel_min_load = st.slider("Minimum load fraction", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    fuel_25 = st.number_input("Fuel use at 25% load (L/h)", min_value=0.0, value=4.5, step=0.1)
    fuel_50 = st.number_input("Fuel use at 50% load (L/h)", min_value=0.0, value=7.4, step=0.1)
    fuel_75 = st.number_input("Fuel use at 75% load (L/h)", min_value=0.0, value=11.0, step=0.1)
    fuel_100 = st.number_input("Fuel use at 100% load (L/h)", min_value=0.0, value=14.7, step=0.1)

    st.header("Battery Datasheet")
    battery_capacity_kwh = st.number_input("Energy capacity (kWh)", min_value=0.0, value=200.0, step=10.0)
    battery_power_kw = st.number_input("Power capacity (kW)", min_value=0.0, value=60.0, step=5.0)
    battery_voltage = st.number_input("Nominal voltage (V)", min_value=1.0, value=48.0, step=1.0)
    battery_charge_eff = st.slider("Charge efficiency", min_value=0.5, max_value=1.0, value=0.93, step=0.01)
    battery_discharge_eff = st.slider("Discharge efficiency", min_value=0.5, max_value=1.0, value=0.93, step=0.01)
    battery_dod = st.slider("Max depth of discharge", min_value=0.1, max_value=1.0, value=0.80, step=0.01)
    battery_k_rate = st.number_input("KiBaM k-rate (1/h)", min_value=0.0, value=0.1, step=0.01, format="%.2f")
    battery_c_fraction = st.slider("Available charge fraction", min_value=0.05, max_value=0.95, value=0.30, step=0.01)

    st.header("Load Inputs")
    base_load_kw = st.number_input("Base load (kW)", min_value=0.0, value=40.0, step=5.0)
    load_type = st.selectbox("Load type", ["residential", "commercial", "industrial"])
    variability_std = st.slider("Load variability", min_value=0.0, max_value=0.3, value=0.08, step=0.01)
    technical_loss = st.slider("Technical loss", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
    non_technical_loss = st.slider("Non-technical loss", min_value=0.0, max_value=0.3, value=0.03, step=0.01)

    st.header("Visual Range")
    days_to_plot = st.slider("Days to show in time-series plots", min_value=1, max_value=max(1, num_days), value=min(14, num_days))

    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)


if run_simulation:
    with st.spinner("Running microgrid simulation..."):
        sim = MicrogridSimulation(
            timestep_minutes=timestep_minutes,
            num_days=num_days,
            start_date=str(start_date),
            pv_capacity_kwp=pv_capacity_kwp,
            wind_capacity_kw=wind_capacity_kw,
            diesel_capacity_kw=diesel_capacity_kw,
            diesel_capacity_kva=diesel_capacity_kva,
            diesel_power_factor=diesel_pf,
            battery_capacity_kwh=battery_capacity_kwh,
            battery_power_kw=battery_power_kw,
            base_load_kw=base_load_kw,
            load_type=load_type,
            dispatch_strategy=dispatch_strategy,
            random_seed=int(random_seed),
            pv_params={
                "temp_coeff_power": pv_temp_coeff,
                "noct": pv_noct,
                "system_losses": pv_losses,
                "inverter_efficiency": pv_inverter_eff,
            },
            wind_params={
                "rated_power_kw": wind_capacity_kw,
                "swept_area_m2": wind_swept_area,
                "hub_height_m": wind_hub_height,
                "cut_in": wind_cut_in,
                "rated_speed": wind_rated_speed,
                "cut_out": wind_cut_out,
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
            },
            battery_params={
                "nominal_voltage": battery_voltage,
                "charge_efficiency": battery_charge_eff,
                "discharge_efficiency": battery_discharge_eff,
                "max_depth_of_discharge": battery_dod,
                "k_rate": battery_k_rate,
                "c_fraction": battery_c_fraction,
            },
            load_params={
                "variability_std": variability_std,
                "technical_loss": technical_loss,
                "non_technical_loss": non_technical_loss,
            },
        )
        results_df = sim.run_simulation(save_results=False, verbose=False)
        metrics = sim.performance_metrics

    st.success("Simulation complete.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Load Served", f"{metrics['load_served_fraction']:.1%}")
    col2.metric("Renewable Fraction", f"{metrics['renewable_fraction']:.1%}")
    col3.metric("Fuel Used", f"{metrics['total_fuel_liters']:.1f} L")
    col4.metric("Cost of Energy", f"${metrics['cost_per_kwh_served']:.3f}/kWh")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Load", f"{metrics['total_load_kwh']:,.0f} kWh")
    col6.metric("Load Shedding", f"{metrics['total_load_shedding_kwh']:,.1f} kWh")
    col7.metric("Avg Battery SOC", f"{metrics['average_battery_soc']:.1f}%")
    col8.metric("Diesel Runtime", f"{metrics['diesel_runtime_hours']:.1f} h")

    tab1, tab2, tab3 = st.tabs(["Visuals", "Results Table", "Downloads"])

    with tab1:
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(build_timeseries_figure(results_df, days_to_plot, sim.steps_per_day), use_container_width=True)
        with right:
            st.pyplot(build_energy_mix_figure(metrics), use_container_width=True)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Total PV generation (kWh)",
                            "Total wind generation (kWh)",
                            "Total diesel generation (kWh)",
                            "Unmet load fraction",
                            "Loss of load probability",
                            "Peak load (kW)",
                        ],
                        "Value": [
                            round(metrics["total_solar_generation_kwh"], 2),
                            round(metrics["total_wind_generation_kwh"], 2),
                            round(metrics["total_diesel_generation_kwh"], 2),
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

else:
    st.info("Set the component datasheet values in the sidebar, then click Run Simulation.")

    st.markdown(
        """
        ### What this app does
        - Accepts user-entered datasheet values for the currently implemented microgrid components.
        - Runs the existing microgrid simulation engine behind the scenes.
        - Presents the outputs as summary indicators, clearer charts, and downloadable tables.

        ### Suggested workflow
        1. Enter component ratings from the datasheets.
        2. Adjust simulation duration and dispatch strategy.
        3. Run the model and compare reliability, renewable fraction, fuel use, and cost.
        """
    )
