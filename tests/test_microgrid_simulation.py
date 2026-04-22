import unittest

from microgrid_simulation import MicrogridSimulation


class MicrogridSimulationSanityTests(unittest.TestCase):
    def test_zero_battery_configuration_runs(self):
        sim = MicrogridSimulation(
            num_days=2,
            timestep_minutes=60,
            pv_capacity_kwp=0.0,
            wind_capacity_kw=0.0,
            hydro_capacity_kw=0.0,
            diesel_capacity_kw=120.0,
            battery_capacity_kwh=0.0,
            battery_power_kw=0.0,
            base_load_kw=40.0,
            load_type="industrial",
            dispatch_strategy="load_following",
            economic_params={"monte_carlo_runs": 0},
            random_seed=3,
        )

        results = sim.run_simulation(save_results=False, verbose=False)
        metrics = sim.performance_metrics

        self.assertEqual(len(results), 48)
        self.assertAlmostEqual(metrics["total_battery_charge_kwh"], 0.0, places=9)
        self.assertAlmostEqual(metrics["total_battery_discharge_kwh"], 0.0, places=9)
        self.assertAlmostEqual(metrics["absolute_energy_balance_error_kwh"], 0.0, places=6)
        self.assertGreater(metrics["total_diesel_generation_kwh"], 0.0)

    def test_energy_balance_closes_for_hybrid_case(self):
        sim = MicrogridSimulation(
            num_days=3,
            timestep_minutes=60,
            pv_capacity_kwp=250.0,
            wind_capacity_kw=50.0,
            hydro_capacity_kw=0.0,
            diesel_capacity_kw=100.0,
            battery_capacity_kwh=250.0,
            battery_power_kw=80.0,
            base_load_kw=45.0,
            load_type="residential",
            dispatch_strategy="economic_dispatch",
            economic_params={"monte_carlo_runs": 0},
            random_seed=5,
        )

        sim.run_simulation(save_results=False, verbose=False)
        metrics = sim.performance_metrics
        supply_balance = (
            metrics["total_generation_kwh"]
            + metrics["total_battery_discharge_kwh"]
            - metrics["total_battery_charge_kwh"]
            - metrics["total_curtailment_kwh"]
            - metrics["total_load_served_kwh"]
        )

        self.assertAlmostEqual(metrics["absolute_energy_balance_error_kwh"], 0.0, places=6)
        self.assertAlmostEqual(supply_balance, 0.0, places=6)
        self.assertGreaterEqual(metrics["load_served_fraction"], 0.99)


if __name__ == "__main__":
    unittest.main()
