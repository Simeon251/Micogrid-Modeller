import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path

from microgrid_simulation import MicrogridSimulation


st.set_page_config(
    page_title="Microgrid Modeler",
    page_icon="M",
    layout="wide",
)


st.markdown(
    """
    <style>
    .kpi-card {
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.5rem;
    }
    .kpi-card .kpi-label {
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 0.35rem;
    }
    .kpi-card .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.15;
        color: #0f172a;
    }
    .kpi-good {
        background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%);
    }
    .kpi-warn {
        background: linear-gradient(180deg, #fff7ed 0%, #fed7aa 100%);
    }
    .kpi-bad {
        background: linear-gradient(180deg, #fef2f2 0%, #fecaca 100%);
    }
    .sidebar-card {
        border-radius: 16px;
        padding: 14px 16px;
        margin: 0.2rem 0 0.8rem 0;
        background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid rgba(37, 99, 235, 0.12);
    }
    .sidebar-card .title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sidebar-card .body {
        font-size: 0.88rem;
        color: #334155;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
        color: #0f172a;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] input,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="base-input"] > div {
        border-radius: 12px !important;
        border-color: rgba(148, 163, 184, 0.45) !important;
        background: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding-top: 0.2rem;
        padding-bottom: 0.35rem;
    }
    [data-testid="stSidebar"] details {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.75);
    }
    [data-testid="stSidebar"] details summary {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_load_served_kpi_class(load_served_fraction):
    if load_served_fraction >= 0.99:
        return "kpi-good"
    if load_served_fraction >= 0.95:
        return "kpi-warn"
    return "kpi-bad"


def get_lcoe_kpi_class(lcoe):
    if lcoe <= 0.40:
        return "kpi-good"
    if lcoe <= 0.70:
        return "kpi-warn"
    return "kpi-bad"


def get_dscr_kpi_class(min_dscr):
    if pd.isna(min_dscr):
        return "kpi-warn"
    if min_dscr >= 1.20:
        return "kpi-good"
    if min_dscr >= 1.00:
        return "kpi-warn"
    return "kpi-bad"


def render_kpi_card(column, label, value, card_class):
    column.markdown(
        f"""
        <div class="kpi-card {card_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_card(title, body):
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="title">{title}</div>
            <div class="body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def persist_uploaded_file(uploaded_file, target_name):
    if uploaded_file is None:
        return None
    upload_dir = Path.cwd() / ".uploaded_inputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / target_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return str(target_path)


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
        colors=["#fde047", "#0ea5e9", "#14b8a6", "#ef4444", "#8b5cf6"],
        alpha=0.85,
    )
    axes[2].plot(x_axis, plot_data["load_kw"], color="black", linewidth=2, linestyle="--", label="Load")
    axes[2].set_title("Dispatch Over Time")
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
    labels = [
        "Direct Renewable",
        "Renewable via Battery",
        "Diesel",
        "Unserved Load",
    ]
    values = [
        metrics["total_direct_renewable_served_kwh"],
        metrics["total_renewable_from_battery_served_kwh"],
        metrics["total_diesel_served_kwh"],
        metrics["total_load_shedding_kwh"],
    ]
    colors = ["#2563eb", "#7c3aed", "#ea580c", "#475569"]

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
    ax.set_title("Served Energy Share")
    fig.tight_layout()
    return fig


def build_served_load_breakdown_figure(metrics):
    labels = ["Served", "Unserved"]
    values = [
        metrics["total_load_served_kwh"],
        metrics["total_load_shedding_kwh"],
    ]
    colors = ["#16a34a", "#dc2626"]

    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    ax.barh(["Load"], [values[0]], color=colors[0], label=labels[0])
    if values[1] > 0:
        ax.barh(["Load"], [values[1]], left=[values[0]], color=colors[1], label=labels[1])
    ax.set_title("Load Service Breakdown")
    ax.set_xlabel("Energy (kWh)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def build_resource_quality_figure(metrics):
    curtailment = metrics["total_curtailment_kwh"]
    renewable_used = metrics["total_renewable_served_kwh"]
    direct_renewable = metrics["total_direct_renewable_served_kwh"]
    renewable_battery = metrics["total_renewable_from_battery_served_kwh"]

    categories = [
        "Direct RE",
        "RE via battery",
        "Curtailment",
    ]
    values = [direct_renewable, renewable_battery, curtailment]
    colors = ["#f59e0b", "#22c55e", "#94a3b8"]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(categories, values, color=colors, alpha=0.9)
    ax.set_title("Renewable Energy Utilization")
    ax.set_ylabel("Energy (kWh)")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
    utilization = renewable_used / max(renewable_used + curtailment, 1e-6)
    ax.text(
        0.02,
        0.95,
        f"Renewable utilization: {utilization:.1%}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    return fig


def build_financial_risk_figure(metrics):
    labels = ["Min DSCR", "Avg DSCR", "LCOE", "LCOE P90"]
    values = [
        metrics["minimum_dscr"] if pd.notna(metrics["minimum_dscr"]) else 0.0,
        metrics["average_dscr"] if pd.notna(metrics["average_dscr"]) else 0.0,
        metrics["lcoe"],
        metrics["lcoe_p90"] if pd.notna(metrics["lcoe_p90"]) else metrics["lcoe"],
    ]
    colors = ["#dc2626" if values[0] < 1.0 else "#16a34a", "#dc2626" if values[1] < 1.0 else "#16a34a", "#2563eb", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.axhline(1.2, color="#b91c1c", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Finance and Risk Snapshot")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def build_generation_totals_figure(metrics):
    labels = ["PV", "Wind", "Hydro", "Diesel", "Battery discharge"]
    values = [
        metrics["total_solar_generation_kwh"],
        metrics["total_wind_generation_kwh"],
        metrics["total_hydro_generation_kwh"],
        metrics["total_diesel_generation_kwh"],
        metrics["total_battery_discharge_kwh"],
    ]
    colors = ["#fde047", "#0ea5e9", "#14b8a6", "#ef4444", "#8b5cf6"]

    filtered = [(label, value, color) for label, value, color in zip(labels, values, colors) if value > 1e-6]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if filtered:
        filtered_labels, filtered_values, filtered_colors = zip(*filtered)
        bars = ax.bar(filtered_labels, filtered_values, color=filtered_colors, alpha=0.9)
        ax.set_ylabel("Energy (kWh)")
        ax.set_title("Generation Totals by Source")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, filtered_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No generation", ha="center", va="center")
        ax.set_title("Generation Totals by Source")
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


def build_model_use_notes(metrics, has_resource_profile, has_load_profile):
    data_mode = metrics.get("resource_data_mode", "synthetic_profiles")
    notes = []
    notes.append(
        f"Resource basis: `{data_mode}`. Measured resource data supports stronger validation than synthetic weather."
    )
    notes.append(
        f"PV performance summary: specific yield is {metrics.get('pv_specific_yield_kwh_per_kwp_year', 0.0):.1f} kWh/kWp-year and solar capacity factor is {metrics.get('solar_capacity_factor', 0.0):.1%}."
    )
    notes.append(
        f"Energy-balance check: absolute balance error is {metrics.get('absolute_energy_balance_error_kwh', 0.0):.3f} kWh over the run, which should stay close to zero."
    )
    if metrics.get("monte_carlo_runs", 0) > 0:
        notes.append(
            f"Risk view: Monte Carlo provides LCOE P50/P90 and DSCR downside metrics across {int(metrics['monte_carlo_runs'])} runs."
        )
    else:
        notes.append(
            "Risk view: Monte Carlo is disabled, so uncertainty is not being quantified in this run."
        )

    validation_checks = [
        "Compare simulated energy, peak load, and fuel use against measured or expected project values.",
        "Run one-at-a-time sensitivities for fuel price, tariff, load growth, PV yield, and battery size.",
        "Use scenarios to compare alternative system designs rather than relying on a single base case.",
    ]
    if not has_resource_profile:
        validation_checks.append(
            "State clearly that solar, wind, and temperature are synthetic because no measured resource CSV was provided."
        )
    if not has_load_profile:
        validation_checks.append(
            "State clearly that demand is synthetic because no measured load CSV was provided."
        )

    limitations = [
        "Synthetic resource profiles are suitable for early-stage screening, not bankable energy assessment.",
        "Dispatch is simplified and does not include all operational constraints of real commercial microgrids.",
        "Financial risk is sampled around key variables, but model-structure uncertainty still remains.",
    ]
    return notes, validation_checks, limitations


def build_financeability_message(metrics):
    min_dscr = metrics.get("minimum_dscr")
    avg_dscr = metrics.get("average_dscr")
    lcoe = metrics.get("lcoe", 0.0)
    tariff_p50 = metrics.get("lcoe_p50")

    if pd.isna(min_dscr) or pd.isna(avg_dscr):
        return "info", "Financeability metrics are unavailable for this run."

    if min_dscr >= 1.20:
        return "success", (
            f"Financeability looks healthy. Minimum DSCR is {min_dscr:.2f}, which is above a common lender threshold of 1.20."
        )
    if min_dscr >= 1.00:
        return "warning", (
            f"Financeability is borderline. Minimum DSCR is {min_dscr:.2f}, so modest changes in tariff, cost, or performance could push the project below debt-service coverage."
        )
    if lcoe > 0 and avg_dscr < 1.0:
        return "error", (
            f"This case appears physically workable but economically unfinanceable under the current assumptions. Minimum DSCR is {min_dscr:.2f} and LCOE is ${lcoe:.3f}/kWh."
        )
    return "warning", (
        f"Debt-service coverage is weak. Minimum DSCR is {min_dscr:.2f}; review tariff, CAPEX, fuel exposure, and reliability assumptions."
    )


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

    render_sidebar_card("Simulation Setup", "Choose the model horizon, dispatch strategy, site, and any measured CSV inputs.")

    setup_col1, setup_col2 = st.columns(2)
    timestep_minutes = setup_col1.selectbox("Timestep (minutes)", [15, 30, 60], index=2, help="Length of each simulation step. Smaller timesteps capture faster system dynamics but take more computation.")
    num_days = setup_col2.slider("Simulation duration (days)", min_value=1, max_value=365, value=30, help="How many days the simulator will model before annualizing the results for lifecycle economics.")
    start_date = setup_col1.date_input("Start date", value=pd.Timestamp("2026-01-01"), help="Calendar date used to build the simulation time index.")
    random_seed = setup_col2.number_input("Random seed", min_value=0, value=42, step=1, help="Keeps synthetic weather and load generation repeatable so you can compare scenarios fairly.")
    dispatch_strategy = st.selectbox("Dispatch strategy", ["economic_dispatch", "load_following", "cycle_charging"], help="`economic_dispatch` chooses the lowest-cost feasible diesel/battery combination each timestep. `load_following` uses diesel only when needed. `cycle_charging` runs diesel at rated output once started and uses surplus to charge the battery.")
    loc_col1, loc_col2 = st.columns(2)
    location_lat = loc_col1.number_input("Latitude", value=-1.94, format="%.4f", help="Used for synthetic solar, wind, and temperature patterns when measured resource CSV data is not provided.")
    location_lon = loc_col2.number_input("Longitude", value=30.06, format="%.4f", help="Used for synthetic resource generation and stored with the scenario assumptions.")
    resource_upload = st.file_uploader("Upload resource CSV", type=["csv"], help="Upload measured or forecast resource data instead of typing a file path.")
    resource_profile_file = st.text_input("Resource profile CSV path", value="", help="Optional path to a timestamped CSV containing measured or forecast resource data such as GHI, wind speed, temperature, hydro flow, and possibly load.")
    load_upload = st.file_uploader("Upload load CSV", type=["csv"], help="Upload measured demand data instead of typing a file path.")
    load_profile_file = st.text_input("Load profile CSV path", value="", help="Optional path to a CSV with a measured load profile. Use this when you want to replace the synthetic demand model.")
    st.info(
        "Choose `economic_dispatch` for lowest operating cost, `load_following` for a renewable-first strategy, or `cycle_charging` when diesel should run harder and charge the battery whenever it turns on."
    )
    st.caption("Tip: Measured CSV data usually gives better results than the synthetic defaults.")
    uploaded_resource_path = persist_uploaded_file(resource_upload, "resource_profile_upload.csv")
    uploaded_load_path = persist_uploaded_file(load_upload, "load_profile_upload.csv")
    effective_resource_profile_file = uploaded_resource_path or (resource_profile_file.strip() or None)
    effective_load_profile_file = uploaded_load_path or (load_profile_file.strip() or None)
    if uploaded_resource_path:
        st.caption(f"Using uploaded resource CSV: `{Path(uploaded_resource_path).name}`")
    if uploaded_load_path:
        st.caption(f"Using uploaded load CSV: `{Path(uploaded_load_path).name}`")

    st.divider()
    render_sidebar_card("Solar PV", "Configure array size, orientation, thermal behavior, and system losses.")
    pv_col1, pv_col2 = st.columns(2)
    pv_capacity_kwp = pv_col1.number_input("Array capacity (kWp)", min_value=0.0, value=100.0, step=10.0, help="Installed DC PV array size at standard test conditions.")
    pv_noct = pv_col2.number_input("NOCT (C)", min_value=0.0, value=46.0, step=1.0, help="Nominal Operating Cell Temperature from the PV module datasheet.")
    pv_temp_coeff = pv_col1.number_input("Temperature coefficient (1/C)", value=-0.00408, format="%.5f", help="Relative power loss per degree Celsius increase in cell temperature above the reference condition.")
    pv_tilt_deg = pv_col2.number_input("Tilt angle (deg)", min_value=0.0, max_value=90.0, value=15.0, step=1.0, help="Panel tilt from horizontal. Adjust this for the actual site and mounting design.")
    pv_azimuth_deg = pv_col1.number_input("Panel azimuth (deg)", min_value=0.0, max_value=360.0, value=0.0, step=5.0, help="Panel facing direction measured clockwise from north. Use 180 deg for south-facing arrays.")
    pv_albedo = st.slider("Ground albedo", min_value=0.0, max_value=1.0, value=0.20, step=0.01, help="Reflectivity of the ground surface used in the irradiance transposition model.")
    pv_losses = st.slider("System losses", min_value=0.0, max_value=0.5, value=0.15, step=0.01, help="Aggregate non-module PV losses such as wiring, mismatch, soiling, and DC collection losses.")
    pv_inverter_eff = st.slider("Inverter efficiency", min_value=0.5, max_value=1.0, value=0.96, step=0.01, help="Fraction of DC PV power converted to AC output.")
    st.caption("Enter the installed PV size and the main module/system performance assumptions from the datasheet.")

    st.divider()
    render_sidebar_card("Wind Turbine", "Use the turbine datasheet or power curve to define aerodynamic performance.")
    wind_col1, wind_col2 = st.columns(2)
    wind_capacity_kw = wind_col1.number_input("Rated power (kW)", min_value=0.0, value=0.0, step=10.0, help="Nameplate AC output of the wind turbine.")
    wind_swept_area = wind_col2.number_input("Swept area (m^2)", min_value=1.0, value=397.6, step=1.0, help="Rotor swept area, usually available from the turbine datasheet.")
    wind_hub_height = wind_col1.number_input("Hub height (m)", min_value=1.0, value=34.0, step=1.0, help="Height of the rotor hub above ground. This affects wind-speed extrapolation.")
    wind_cut_in = wind_col2.number_input("Cut-in speed (m/s)", min_value=0.0, value=3.5, step=0.1, help="Minimum wind speed at which the turbine starts producing useful power.")
    wind_rated_speed = wind_col1.number_input("Rated speed (m/s)", min_value=0.1, value=10.5, step=0.1, help="Wind speed at which the turbine reaches rated output.")
    wind_cut_out = wind_col2.number_input("Cut-out speed (m/s)", min_value=0.1, value=20.0, step=0.1, help="Wind speed above which the turbine shuts down for protection.")
    st.caption("Use the turbine datasheet or power curve for these values, especially the cut-in, rated, and cut-out speeds.")

    st.divider()
    render_sidebar_card("Hydropower", "Set plant size, hydraulic design, and environmental flow limits.")
    hydro_col1, hydro_col2 = st.columns(2)
    hydro_capacity_kw = hydro_col1.number_input("Hydro rated power (kW)", min_value=0.0, value=0.0, step=10.0, help="Nameplate output of the hydropower plant or turbine.")
    hydro_design_flow_m3s = hydro_col2.number_input("Design flow (m^3/s)", min_value=0.0, value=1.0, step=0.1, help="Flow level the hydro turbine is designed to use at rated conditions.")
    hydro_head_m = hydro_col1.number_input("Net head (m)", min_value=0.0, value=20.0, step=1.0, help="Effective head at the turbine after hydraulic losses.")
    hydro_efficiency = st.slider("Hydro efficiency", min_value=0.1, max_value=1.0, value=0.85, step=0.01, help="Overall turbine-generator conversion efficiency.")
    hydro_min_flow_fraction = st.slider("Hydro minimum flow fraction", min_value=0.0, max_value=1.0, value=0.20, step=0.01, help="Minimum fraction of design flow needed before the hydro unit can operate.")
    hydro_environmental_flow_m3s = st.number_input("Environmental flow (m^3/s)", min_value=0.0, value=0.0, step=0.1, help="Flow that must remain in the river and cannot be diverted to generation.")
    st.caption("If you have a measured river-flow dataset, use it through the resource CSV path for a much more realistic hydro simulation.")

    st.divider()
    render_sidebar_card("Diesel Generator", "Set generator rating, fuel curve, and optional reliability behavior.")
    diesel_use_kva = st.toggle("Input diesel in kVA", value=True)
    if diesel_use_kva:
        diesel_capacity_kva = st.number_input("Generator rating (kVA)", min_value=0.0, value=60.0, step=5.0, help="Apparent power rating from the generator datasheet.")
        diesel_capacity_kw = 0.0
    else:
        diesel_capacity_kw = st.number_input("Prime rating (kW)", min_value=0.0, value=48.0, step=5.0, help="Real power rating if your datasheet already gives the generator in kW.")
        diesel_capacity_kva = None
    diesel_pf = st.slider("Power factor", min_value=0.5, max_value=1.0, value=0.8, step=0.01, help="Used to convert kVA into kW when the generator is specified in apparent power.")
    diesel_min_load = st.slider("Minimum load fraction", min_value=0.0, max_value=1.0, value=0.25, step=0.01, help="Minimum stable loading fraction. Below this, real engines often operate poorly or should not run continuously.")
    fuel_col1, fuel_col2 = st.columns(2)
    fuel_25 = fuel_col1.number_input("Fuel use at 25% load (L/h)", min_value=0.0, value=4.5, step=0.1, help="Fuel use from the generator datasheet at 25% loading.")
    fuel_50 = fuel_col2.number_input("Fuel use at 50% load (L/h)", min_value=0.0, value=7.4, step=0.1, help="Fuel use from the generator datasheet at 50% loading.")
    fuel_75 = fuel_col1.number_input("Fuel use at 75% load (L/h)", min_value=0.0, value=11.0, step=0.1, help="Fuel use from the generator datasheet at 75% loading.")
    fuel_100 = fuel_col2.number_input("Fuel use at 100% load (L/h)", min_value=0.0, value=14.7, step=0.1, help="Fuel use from the generator datasheet at full load.")
    st.caption("Fuel values should come from the generator fuel-consumption table at different loading points.")

    with st.expander("Advanced: Diesel Reliability", expanded=False):
        enable_generator_reliability = st.toggle("Enable diesel reliability modeling", value=False, help="Adds planned maintenance and stochastic forced outages using MTBF and MTTR assumptions.")
        rel_col1, rel_col2 = st.columns(2)
        mtbf_hours = rel_col1.number_input("Diesel MTBF (hours)", min_value=1.0, value=500.0, step=50.0, help="Average operating time between forced failures.")
        mttr_hours = rel_col2.number_input("Diesel MTTR (hours)", min_value=1.0, value=8.0, step=1.0, help="Average repair duration after a forced outage.")
        planned_maintenance_interval_hours = rel_col1.number_input("Planned maintenance interval (runtime-hours)", min_value=1.0, value=1000.0, step=100.0, help="Runtime interval between planned maintenance shutdowns.")
        planned_maintenance_duration_hours = rel_col2.number_input("Planned maintenance duration (hours)", min_value=1.0, value=6.0, step=1.0, help="Length of each planned maintenance event.")

    st.divider()
    render_sidebar_card("Battery", "Define storage size, efficiency, and degradation assumptions.")
    batt_col1, batt_col2 = st.columns(2)
    battery_capacity_kwh = batt_col1.number_input("Energy capacity (kWh)", min_value=0.0, value=200.0, step=10.0, help="Total usable battery energy capacity.")
    battery_power_kw = batt_col2.number_input("Power capacity (kW)", min_value=0.0, value=60.0, step=5.0, help="Maximum battery charge or discharge power.")
    battery_voltage = batt_col1.number_input("Nominal voltage (V)", min_value=1.0, value=48.0, step=1.0, help="Nominal DC battery voltage used in the KiBaM formulation.")
    battery_charge_eff = st.slider("Charge efficiency", min_value=0.5, max_value=1.0, value=0.93, step=0.01, help="Fraction of charging energy that is stored in the battery.")
    battery_discharge_eff = st.slider("Discharge efficiency", min_value=0.5, max_value=1.0, value=0.93, step=0.01, help="Fraction of stored energy that can be delivered back to the system.")
    battery_dod = st.slider("Max depth of discharge", min_value=0.1, max_value=1.0, value=0.80, step=0.01, help="Maximum share of the battery capacity that the controller is allowed to use.")
    battery_k_rate = batt_col2.number_input("KiBaM k-rate (1/h)", min_value=0.0, value=0.1, step=0.01, format="%.2f", help="Charge transfer rate between available and bound charge reservoirs in the KiBaM battery model.")
    battery_c_fraction = st.slider("Available charge fraction", min_value=0.05, max_value=0.95, value=0.30, step=0.01, help="Fraction of total battery charge that is immediately available in the KiBaM model.")
    battery_degradation_temp_sensitivity = st.number_input("Battery degradation temp sensitivity", min_value=0.0, value=0.025, step=0.005, format="%.3f", help="Multiplier that increases battery fade above 25 C.")
    battery_eol_fraction = st.slider("Battery end-of-life fraction", min_value=0.50, max_value=0.95, value=0.80, step=0.01, help="Battery replacement threshold as remaining capacity fraction.")
    st.caption("If you are unsure about `k-rate` or `available charge fraction`, keep the defaults unless you have calibration data.")

    st.divider()
    render_sidebar_card("Load Inputs", "Describe demand level, variability, losses, and optional flexibility.")
    load_col1, load_col2 = st.columns(2)
    base_load_kw = load_col1.number_input("Base load (kW)", min_value=0.0, value=40.0, step=5.0, help="Average demand anchor for the synthetic load generator.")
    load_type = load_col2.selectbox("Load type", ["residential", "commercial", "industrial"], help="Selects a built-in daily and weekly demand pattern when no measured load CSV is provided.")
    variability_std = st.slider("Load variability", min_value=0.0, max_value=0.3, value=0.08, step=0.01, help="Controls stochastic variability around the base synthetic demand pattern.")
    price_elasticity = st.slider("Load price elasticity", min_value=-1.0, max_value=0.0, value=0.0, step=0.05, help="Negative values reduce load when tariff multipliers rise.")
    tariff_multiplier = st.slider("Tariff multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05, help="Relative retail tariff applied to the synthetic load model.")
    technical_loss = st.slider("Technical loss", min_value=0.0, max_value=0.3, value=0.05, step=0.01, help="Distribution and conversion losses added on top of useful load.")
    non_technical_loss = st.slider("Non-technical loss", min_value=0.0, max_value=0.3, value=0.03, step=0.01, help="Commercial or non-metered losses added on top of useful load.")
    st.caption("These fields matter only when you are using the built-in synthetic load model instead of a measured load CSV.")

    with st.expander("Advanced: Demand-Side Management", expanded=False):
        enable_dsm = st.toggle("Enable DSM", value=False, help="Applies simple load shifting and peak shaving to make demand more flexible.")
        deferrable_load_fraction = st.slider("Deferrable load fraction", min_value=0.0, max_value=0.6, value=0.15, step=0.01, help="Share of peak-period demand that can be shifted to another time window.")
        peak_reduction_fraction = st.slider("Peak reduction fraction", min_value=0.0, max_value=0.5, value=0.05, step=0.01, help="Additional peak-period demand that is curtailed rather than shifted.")
        dsm_col1, dsm_col2 = st.columns(2)
        peak_start_hour = dsm_col1.slider("Peak start hour", min_value=0, max_value=23, value=18, step=1, help="Start of the demand peak window used by DSM.")
        peak_end_hour = dsm_col2.slider("Peak end hour", min_value=1, max_value=24, value=22, step=1, help="End of the demand peak window used by DSM.")
        shift_start_hour = dsm_col1.slider("Shift-to start hour", min_value=0, max_value=23, value=10, step=1, help="Start of the preferred window that receives shifted demand.")
        shift_end_hour = dsm_col2.slider("Shift-to end hour", min_value=1, max_value=24, value=16, step=1, help="End of the preferred load-shifting window.")

    st.divider()
    render_sidebar_card("Economics and Risk", "Review tariff, CAPEX, O&M, financing, and uncertainty assumptions.")
    econ_col1, econ_col2 = st.columns(2)
    project_lifetime_years = econ_col1.slider("Project life (years)", min_value=1, max_value=30, value=20, help="Years included in the lifecycle cost and LCOE calculation.")
    nominal_discount_rate = econ_col2.slider("Discount rate", min_value=0.0, max_value=0.3, value=0.12, step=0.01, help="Nominal discount rate used to bring future costs and energy to present value.")
    fuel_price_per_liter = econ_col1.number_input("Fuel price ($/L)", min_value=0.0, value=1.50, step=0.05, help="Current diesel fuel price used in dispatch and lifecycle economics.")
    fuel_price_escalation_rate = st.slider("Fuel price escalation", min_value=0.0, max_value=0.2, value=0.05, step=0.01, help="Expected annual growth in fuel price over the project lifetime.")
    om_escalation_rate = st.slider("O&M escalation", min_value=0.0, max_value=0.2, value=0.03, step=0.01, help="Expected annual increase in operations and maintenance costs.")
    energy_tariff_per_kwh = econ_col2.number_input("Energy tariff ($/kWh)", min_value=0.0, value=0.75, step=0.01, help="Average revenue tariff used for DSCR and project cash flow.")
    tariff_escalation_rate = st.slider("Tariff escalation", min_value=0.0, max_value=0.2, value=0.03, step=0.01, help="Expected annual growth in retail tariff.")
    unserved_energy_cost_per_kwh = st.number_input("Unserved energy penalty ($/kWh)", min_value=0.0, value=2.0, step=0.1, help="Economic penalty assigned to each kWh of unmet demand.")
    st.caption("These values drive lifecycle cost and LCOE, so use assumptions that match your project finance context.")

    with st.expander("Advanced: Capital Cost Assumptions", expanded=False):
        capex_col1, capex_col2 = st.columns(2)
        pv_capex_per_kwp = capex_col1.number_input("PV CAPEX ($/kWp)", min_value=0.0, value=900.0, step=25.0, help="Installed capital cost per kWp of PV.")
        wind_capex_per_kw = capex_col2.number_input("Wind CAPEX ($/kW)", min_value=0.0, value=1500.0, step=50.0, help="Installed capital cost per kW of wind capacity.")
        hydro_capex_per_kw = capex_col1.number_input("Hydro CAPEX ($/kW)", min_value=0.0, value=2500.0, step=50.0, help="Installed capital cost per kW of hydropower capacity.")
        diesel_capex_per_kw = capex_col2.number_input("Diesel CAPEX ($/kW)", min_value=0.0, value=550.0, step=25.0, help="Installed capital cost per kW of diesel generator capacity.")
        battery_capex_per_kwh = capex_col1.number_input("Battery CAPEX ($/kWh)", min_value=0.0, value=350.0, step=10.0, help="Installed battery energy cost per kWh.")
        battery_power_capex_per_kw = capex_col2.number_input("Battery power CAPEX ($/kW)", min_value=0.0, value=150.0, step=10.0, help="Additional battery power electronics and converter cost per kW.")

    with st.expander("Advanced: O&M, Debt and Risk", expanded=False):
        om_col1, om_col2 = st.columns(2)
        pv_fixed_om_per_kw_year = om_col1.number_input("PV fixed O&M ($/kW-yr)", min_value=0.0, value=18.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of PV.")
        wind_fixed_om_per_kw_year = om_col2.number_input("Wind fixed O&M ($/kW-yr)", min_value=0.0, value=45.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of wind.")
        hydro_fixed_om_per_kw_year = om_col1.number_input("Hydro fixed O&M ($/kW-yr)", min_value=0.0, value=35.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of hydropower.")
        diesel_fixed_om_per_kw_year = om_col2.number_input("Diesel fixed O&M ($/kW-yr)", min_value=0.0, value=20.0, step=1.0, help="Annual fixed operations and maintenance cost per kW of diesel capacity.")
        diesel_maintenance_cost_per_hour = om_col1.number_input("Diesel maintenance ($/runtime-hour)", min_value=0.0, value=1.5, step=0.1, help="Maintenance cost applied to each generator runtime hour.")
        battery_fixed_om_per_kwh_year = om_col2.number_input("Battery fixed O&M ($/kWh-yr)", min_value=0.0, value=8.0, step=1.0, help="Annual fixed operations and maintenance cost per kWh of installed battery energy.")
        diesel_variable_om_per_kwh = om_col1.number_input("Diesel variable O&M ($/kWh)", min_value=0.0, value=0.03, step=0.01, format="%.2f", help="Variable cost applied to each kWh produced by the diesel generator.")
        battery_variable_om_per_kwh = om_col2.number_input("Battery variable O&M ($/kWh)", min_value=0.0, value=0.01, step=0.01, format="%.2f", help="Variable cost applied to each kWh discharged from the battery.")
        debt_fraction = st.slider("Debt fraction", min_value=0.0, max_value=0.95, value=0.70, step=0.05, help="Share of upfront CAPEX financed by debt.")
        debt_interest_rate = st.slider("Debt interest rate", min_value=0.0, max_value=0.25, value=0.10, step=0.01, help="Nominal annual debt interest rate.")
        debt_tenor_years = st.slider("Debt tenor (years)", min_value=1, max_value=20, value=10, help="Years over which the model repays debt.")
        monte_carlo_runs = st.slider("Monte Carlo runs", min_value=0, max_value=1000, value=200, step=50, help="Number of stochastic finance scenarios used for risk outputs.")
        risk_col1, risk_col2 = st.columns(2)
        fuel_price_volatility = risk_col1.slider("Fuel price volatility", min_value=0.0, max_value=0.6, value=0.18, step=0.01, help="Annualized fuel price volatility used in the GBM process.")
        inflation_volatility = risk_col2.slider("Inflation volatility", min_value=0.0, max_value=0.2, value=0.02, step=0.005, help="Shock size for annual CPI in the AR(1) process.")
        exchange_rate_volatility = risk_col1.slider("FX volatility", min_value=0.0, max_value=0.4, value=0.08, step=0.01, help="Shock size for annual FX in the AR(1) process.")

    st.divider()
    render_sidebar_card("Visual Range", "Choose how much of the simulation period to show in the dashboard plots.")
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
            load_profile_file=effective_load_profile_file,
            resource_profile_file=effective_resource_profile_file,
            dispatch_strategy=dispatch_strategy,
            random_seed=int(random_seed),
            pv_params={
                "temp_coeff_power": pv_temp_coeff,
                "noct": pv_noct,
                "tilt_deg": pv_tilt_deg,
                "panel_azimuth_deg": pv_azimuth_deg,
                "albedo": pv_albedo,
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
        model_notes, validation_checks, limitations = build_model_use_notes(
            metrics,
            has_resource_profile=bool(effective_resource_profile_file),
            has_load_profile=bool(effective_load_profile_file),
        )
        financeability_level, financeability_message = build_financeability_message(metrics)

    st.success("Simulation complete.")

    st.subheader("Headline KPIs")
    col1, col2, col3, col4 = st.columns(4)
    render_kpi_card(
        col1,
        "Load Served",
        f"{metrics['load_served_fraction']:.1%}",
        get_load_served_kpi_class(metrics["load_served_fraction"]),
    )
    col2.metric("Renewable Fraction", f"{metrics['renewable_fraction']:.1%}")
    col3.metric("Fuel Used", f"{metrics['total_fuel_liters']:.1f} L")
    render_kpi_card(
        col4,
        "LCOE",
        f"${metrics['lcoe']:.3f}/kWh",
        get_lcoe_kpi_class(metrics["lcoe"]),
    )

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Battery SOC", f"{metrics['average_battery_soc']:.1f}%")
    col6.metric("Diesel Runtime", f"{metrics['diesel_runtime_hours']:.1f} h")
    render_kpi_card(
        col7,
        "Min DSCR",
        f"{metrics['minimum_dscr']:.2f}" if pd.notna(metrics["minimum_dscr"]) else "N/A",
        get_dscr_kpi_class(metrics["minimum_dscr"]),
    )
    col8.metric("Resource Mode", metrics["resource_data_mode"].replace("_", " ").title())

    if financeability_level == "success":
        st.success(financeability_message)
    elif financeability_level == "warning":
        st.warning(financeability_message)
    elif financeability_level == "error":
        st.error(financeability_message)
    else:
        st.info(financeability_message)

    st.subheader("Visual Summary")
    vis_left, vis_right = st.columns([1, 1])
    with vis_left:
        st.pyplot(build_served_load_breakdown_figure(metrics), use_container_width=True)
        st.pyplot(build_energy_mix_figure(metrics), use_container_width=True)
    with vis_right:
        st.pyplot(build_resource_quality_figure(metrics), use_container_width=True)
        st.pyplot(build_financial_risk_figure(metrics), use_container_width=True)

    st.subheader("Key Supporting Metrics")
    metric_table_left, metric_table_right = st.columns([1, 1])
    with metric_table_left:
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Total load (kWh)",
                        "Load shedding (kWh)",
                        "Peak load after DSM (kW)",
                        "Peak load before DSM (kW)",
                        "Diesel availability",
                        "Battery replacement interval (yr)",
                        "PV specific yield (kWh/kWp-yr)",
                        "Solar capacity factor",
                    ],
                    "Value": [
                        round(metrics["total_load_kwh"], 1),
                        round(metrics["total_load_shedding_kwh"], 1),
                        round(metrics["peak_load_kw"], 2),
                        round(metrics["peak_baseline_load_kw"], 2),
                        f"{metrics['diesel_availability_fraction']:.1%}",
                        round(metrics["battery_replacement_interval_years"], 2),
                        round(metrics["pv_specific_yield_kwh_per_kwp_year"], 1),
                        f"{metrics['solar_capacity_factor']:.1%}",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with metric_table_right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": [
                        "Upfront CAPEX ($)",
                        "Lifecycle cost NPV ($)",
                        "Operating cost per kWh ($/kWh)",
                        "LCOE P50 ($/kWh)",
                        "LCOE P90 ($/kWh)",
                        "Curtailment (kWh)",
                        "DSM shifted energy (kWh)",
                        "Energy balance error (kWh)",
                    ],
                    "Value": [
                        round(metrics["upfront_capex"], 2),
                        round(metrics["discounted_lifecycle_cost"], 2),
                        round(metrics["operating_cost_per_kwh_served"], 4),
                        round(metrics["lcoe_p50"], 4) if pd.notna(metrics["lcoe_p50"]) else "N/A",
                        round(metrics["lcoe_p90"], 4) if pd.notna(metrics["lcoe_p90"]) else "N/A",
                        round(metrics["total_curtailment_kwh"], 1),
                        round(metrics["total_dsm_shifted_energy_kwh"], 1),
                        round(metrics["absolute_energy_balance_error_kwh"], 6),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

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
            st.pyplot(build_generation_totals_figure(metrics), use_container_width=True)
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

    with st.expander("Using Your Model: Presentation Notes", expanded=False):
        st.markdown("**Key points to report**")
        for item in model_notes:
            st.write(f"- {item}")

        st.markdown("**Validation and sensitivity checklist**")
        for item in validation_checks:
            st.write(f"- {item}")

        st.markdown("**Limitations to state clearly**")
        for item in limitations:
            st.write(f"- {item}")

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
