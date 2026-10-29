"""Wall-time following must remain bounded and recordings must replay exactly."""
import unittest
import numpy as np

from single_arm_runtime import PhysicsClock,recorded_step_durations,runtime_summary


class PhysicsClockTests(unittest.TestCase):
    def test_slow_ui_does_not_turn_simulation_into_fixed_tick_slow_motion(self):
        clock=PhysicsClock(.0005)
        elapsed=[.05,.08,.07,.09,.06]
        steps=[clock.advance(dt) for dt in elapsed]
        self.assertAlmostEqual(sum(steps),sum(elapsed),places=10)
        self.assertAlmostEqual(clock.unserved_s,0.)
        self.assertGreater(steps[1],.05)

    def test_long_stall_is_bounded_and_time_debt_is_reported_not_replayed(self):
        clock=PhysicsClock(.0005)
        self.assertEqual(clock.advance(2.),.1)
        self.assertAlmostEqual(clock.unserved_s,1.9)
        self.assertEqual(clock.advance(.05),.05)

    def test_fractional_substeps_do_not_accumulate_rounding_drift(self):
        clock=PhysicsClock(.0005)
        simulated=sum(clock.advance(.05023) for _ in range(200))
        self.assertLess(abs(simulated-200*.05023),.0005)

    def test_old_recordings_keep_the_original_fixed_interval(self):
        durations=recorded_step_durations({},dict(control_hz=20),3)
        np.testing.assert_array_equal(durations,[.05]*3)

    def test_variable_recording_requires_finite_positive_integral_substeps(self):
        manifest=dict(control_hz=20,timestep_s=.0005,physics_clock='bounded_wall_time')
        for steps in ([.05,float('nan'),.05],[.05,-.05,.05],[.05,.05025,.05]):
            with self.assertRaises(ValueError):recorded_step_durations(dict(physics_dt_s=np.array(steps)),manifest,3)
        with self.assertRaises(ValueError):recorded_step_durations({},manifest,3)
        np.testing.assert_array_equal(recorded_step_durations(dict(physics_dt_s=np.array([.05,.08,.06])),manifest,3),[.05,.08,.06])

    def test_runtime_report_uses_actual_host_intervals(self):
        rows=[dict(host_tick_ns=i*80_000_000,sim_time_s=i*.08,control_work_ms=65.,unserved_wall_s=0.) for i in range(3)]
        report=runtime_summary(rows)
        self.assertAlmostEqual(report['achieved_control_hz'],12.5)
        self.assertAlmostEqual(report['realtime_factor'],1.)
        self.assertEqual(report['overrun_rows'],3)


if __name__=='__main__':unittest.main()
