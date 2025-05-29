import unittest
import numpy as np
from descent import LunarDescentGuidance


class TestLunarDescentGuidance(unittest.TestCase):
    """Test cases for the LunarDescentGuidance class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.guidance = LunarDescentGuidance()

    def test_initialization_default_values(self):
        """Test that the guidance system initializes with correct default values"""
        self.assertEqual(self.guidance.dt, 0.1)
        self.assertEqual(self.guidance.tau_vert, 5.0)
        self.assertEqual(self.guidance.program_number, 0)
        self.assertEqual(self.guidance.wch_phase, 0)
        self.assertEqual(self.guidance.wch_vert, 0)
        self.assertFalse(self.guidance.x_override)

        # Check numpy arrays
        np.testing.assert_array_equal(self.guidance.lunar_gravity, [0.0, 0.0, -1.622])
        np.testing.assert_array_equal(self.guidance.r_gu, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(self.guidance.v_gu, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(self.guidance.v2fg, [0.0, 0.0, -1.22])
        np.testing.assert_array_equal(self.guidance.g_eff, [0.0, 0.0, -1.622])
        np.testing.assert_array_equal(self.guidance.a_command, [0.0, 0.0, 0.0])

    def test_initialization_display_variables(self):
        """Test that display variables are initialized correctly"""
        self.assertEqual(self.guidance.altitude, 0.0)
        self.assertEqual(self.guidance.alt_rate, 0.0)
        self.assertEqual(self.guidance.hor_velocity, 0.0)

    def test_p65_start_initialization(self):
        """Test P65 start sequence sets correct values"""
        self.guidance.p65_start()

        self.assertEqual(self.guidance.program_number, 65)
        self.assertEqual(self.guidance.wch_vert, -2)
        self.assertTrue(self.guidance.x_override)
        self.assertEqual(self.guidance.wch_phase, 2)

    def test_update_state_position_only(self):
        """Test updating state with position vector"""
        test_position = [10.0, 20.0, -30.0]
        test_velocity = [1.0, 2.0, -3.0]

        self.guidance.update_state(test_position, test_velocity)

        np.testing.assert_array_equal(self.guidance.r_gu, test_position)
        np.testing.assert_array_equal(self.guidance.v_gu, test_velocity)

    def test_update_state_with_numpy_arrays(self):
        """Test updating state with numpy arrays"""
        test_position = np.array([5.0, -10.0, 15.0])
        test_velocity = np.array([-2.0, 3.0, 1.0])

        self.guidance.update_state(test_position, test_velocity)

        np.testing.assert_array_equal(self.guidance.r_gu, test_position)
        np.testing.assert_array_equal(self.guidance.v_gu, test_velocity)

    def test_set_descent_rate_positive(self):
        """Test setting positive descent rate"""
        self.guidance.set_descent_rate(2.0)
        self.assertEqual(self.guidance.v2fg[2], -2.0)

    def test_set_descent_rate_zero(self):
        """Test setting zero descent rate (hover)"""
        self.guidance.set_descent_rate(0.0)
        self.assertEqual(self.guidance.v2fg[2], 0.0)

    def test_set_descent_rate_negative(self):
        """Test setting negative descent rate (ascent)"""
        self.guidance.set_descent_rate(-1.5)
        self.assertEqual(self.guidance.v2fg[2], 1.5)

    def test_p65_guidance_zero_velocity_error(self):
        """Test guidance when current velocity matches desired velocity"""
        # Set current velocity to match desired velocity
        self.guidance.v_gu = np.array([0.0, 0.0, -1.22])
        self.guidance.v2fg = np.array([0.0, 0.0, -1.22])

        accel_command = self.guidance.p65_guidance()

        # Should command acceleration to counteract gravity only
        expected_accel = -self.guidance.g_eff
        np.testing.assert_array_almost_equal(accel_command, expected_accel)

    def test_p65_guidance_with_velocity_error(self):
        """Test guidance with velocity error"""
        # Current velocity different from desired
        self.guidance.v_gu = np.array([1.0, 0.5, -2.0])
        self.guidance.v2fg = np.array([0.0, 0.0, -1.22])

        accel_command = self.guidance.p65_guidance()

        # Calculate expected acceleration
        velocity_error = self.guidance.v_gu - self.guidance.v2fg
        expected_accel = velocity_error / self.guidance.tau_vert - self.guidance.g_eff

        np.testing.assert_array_almost_equal(accel_command, expected_accel)

    def test_p65_guidance_updates_display_variables(self):
        """Test that guidance updates altitude, alt_rate, and horizontal velocity"""
        # Set test position and velocity
        test_position = np.array([3.0, 4.0, -50.0])  # 50m altitude
        test_velocity = np.array([1.0, 2.0, -1.5])   # 1.5 m/s descent, sqrt(5) horizontal

        self.guidance.update_state(test_position, test_velocity)
        self.guidance.p65_guidance()

        self.assertAlmostEqual(self.guidance.altitude, 50.0)
        self.assertAlmostEqual(self.guidance.alt_rate, 1.5)
        self.assertAlmostEqual(self.guidance.hor_velocity, np.sqrt(5.0), places=5)

    def test_altitude_calculation_positive_z(self):
        """Test altitude calculation with positive Z position"""
        self.guidance.r_gu = np.array([0.0, 0.0, 100.0])
        self.guidance.p65_guidance()
        self.assertEqual(self.guidance.altitude, -100.0)  # Negative altitude (below surface)

    def test_altitude_calculation_negative_z(self):
        """Test altitude calculation with negative Z position"""
        self.guidance.r_gu = np.array([0.0, 0.0, -75.0])
        self.guidance.p65_guidance()
        self.assertEqual(self.guidance.altitude, 75.0)  # Positive altitude (above surface)

    def test_horizontal_velocity_calculation(self):
        """Test horizontal velocity magnitude calculation"""
        test_cases = [
            ([3.0, 4.0, -1.0], 5.0),  # 3-4-5 triangle
            ([0.0, 0.0, -2.0], 0.0),  # No horizontal velocity
            ([1.0, 1.0, -1.0], np.sqrt(2.0)),  # Equal X and Y components
            ([-2.0, 2.0, -1.0], np.sqrt(8.0))   # Negative X component
        ]

        for velocity, expected_hor_vel in test_cases:
            with self.subTest(velocity=velocity):
                self.guidance.v_gu = np.array(velocity)
                self.guidance.p65_guidance()
                self.assertAlmostEqual(self.guidance.hor_velocity, expected_hor_vel, places=5)

    def test_check_mode_switch_manual_throttle(self):
        """Test mode switch to P67 when manual throttle is active"""
        self.guidance.program_number = 65
        result = self.guidance.check_mode_switch(manual_throttle=True, rod_switch_clicked=False)
        self.assertEqual(result, 67)

    def test_check_mode_switch_rod_switch(self):
        """Test mode switch to P66 when ROD switch is clicked"""
        self.guidance.program_number = 65
        result = self.guidance.check_mode_switch(manual_throttle=False, rod_switch_clicked=True)
        self.assertEqual(result, 66)

    def test_check_mode_switch_both_conditions(self):
        """Test mode switch priority when both conditions are true"""
        self.guidance.program_number = 65
        # Manual throttle should take priority over ROD switch
        result = self.guidance.check_mode_switch(manual_throttle=True, rod_switch_clicked=True)
        self.assertEqual(result, 67)

    def test_check_mode_switch_no_conditions(self):
        """Test no mode switch when no conditions are met"""
        self.guidance.program_number = 65
        result = self.guidance.check_mode_switch(manual_throttle=False, rod_switch_clicked=False)
        self.assertEqual(result, 65)

    def test_guidance_with_extreme_velocity_error(self):
        """Test guidance behavior with large velocity errors"""
        # Set very large velocity error
        self.guidance.v_gu = np.array([10.0, -5.0, 5.0])
        self.guidance.v2fg = np.array([0.0, 0.0, -1.22])

        accel_command = self.guidance.p65_guidance()

        # Verify the acceleration command is calculated correctly
        velocity_error = np.array([10.0, -5.0, 6.22])
        expected_accel = velocity_error / 5.0 - self.guidance.g_eff
        np.testing.assert_array_almost_equal(accel_command, expected_accel)

    def test_guidance_with_different_tau_vert(self):
        """Test guidance behavior with different time constants"""
        self.guidance.tau_vert = 10.0  # Slower response
        self.guidance.v_gu = np.array([2.0, 0.0, -2.0])
        self.guidance.v2fg = np.array([0.0, 0.0, -1.22])

        accel_command = self.guidance.p65_guidance()

        velocity_error = np.array([2.0, 0.0, -0.78])
        expected_accel = velocity_error / 10.0 - self.guidance.g_eff
        np.testing.assert_array_almost_equal(accel_command, expected_accel)

    def test_multiple_guidance_cycles(self):
        """Test multiple consecutive guidance cycles"""
        positions = [
            np.array([0.0, 0.0, -100.0]),
            np.array([1.0, 0.5, -95.0]),
            np.array([2.0, 1.0, -90.0])
        ]
        velocities = [
            np.array([1.0, 0.5, -1.0]),
            np.array([1.0, 0.5, -1.1]),
            np.array([1.0, 0.5, -1.2])
        ]

        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            with self.subTest(cycle=i):
                self.guidance.update_state(pos, vel)
                accel_command = self.guidance.p65_guidance()

                # Verify acceleration command is computed
                self.assertIsInstance(accel_command, np.ndarray)
                self.assertEqual(len(accel_command), 3)

                # Verify display variables are updated
                self.assertEqual(self.guidance.altitude, -pos[2])
                self.assertEqual(self.guidance.alt_rate, -vel[2])

    def test_descent_rate_boundary_values(self):
        """Test setting descent rate with boundary values"""
        test_rates = [0.001, 0.1, 1.0, 5.0, 10.0, 100.0]

        for rate in test_rates:
            with self.subTest(rate=rate):
                self.guidance.set_descent_rate(rate)
                self.assertEqual(self.guidance.v2fg[2], -rate)

    def test_state_persistence_across_calls(self):
        """Test that state persists correctly across multiple function calls"""
        # Set initial state
        initial_pos = np.array([5.0, 10.0, -20.0])
        initial_vel = np.array([0.5, -0.3, -1.5])

        self.guidance.update_state(initial_pos, initial_vel)

        # Run guidance
        self.guidance.p65_guidance()

        # Verify state is preserved
        np.testing.assert_array_equal(self.guidance.r_gu, initial_pos)
        np.testing.assert_array_equal(self.guidance.v_gu, initial_vel)

        # Update state again
        new_pos = np.array([5.1, 9.9, -19.8])
        new_vel = np.array([0.4, -0.2, -1.4])

        self.guidance.update_state(new_pos, new_vel)

        # Verify new state is stored
        np.testing.assert_array_equal(self.guidance.r_gu, new_pos)
        np.testing.assert_array_equal(self.guidance.v_gu, new_vel)

    def test_acceleration_command_storage(self):
        """Test that acceleration commands are properly stored"""
        self.guidance.v_gu = np.array([1.0, 2.0, -3.0])

        accel_command = self.guidance.p65_guidance()

        # Verify the returned command matches stored command
        np.testing.assert_array_equal(accel_command, self.guidance.a_command)

    def test_gravity_effect_on_guidance(self):
        """Test that gravity is properly accounted for in guidance"""
        # Set zero velocity error (current = desired)
        self.guidance.v_gu = np.array([0.0, 0.0, -1.22])
        self.guidance.v2fg = np.array([0.0, 0.0, -1.22])

        accel_command = self.guidance.p65_guidance()

        # Should command upward acceleration to counteract downward gravity
        expected_accel = np.array([0.0, 0.0, 1.622])
        np.testing.assert_array_almost_equal(accel_command, expected_accel)

    def test_display_variables_with_zero_values(self):
        """Test display variables when position and velocity are zero"""
        self.guidance.update_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.guidance.p65_guidance()

        self.assertEqual(self.guidance.altitude, 0.0)
        self.assertEqual(self.guidance.alt_rate, 0.0)
        self.assertEqual(self.guidance.hor_velocity, 0.0)

    def test_p65_start_multiple_calls(self):
        """Test that multiple calls to p65_start maintain correct state"""
        # Call p65_start multiple times
        for i in range(3):
            with self.subTest(call=i):
                self.guidance.p65_start()

                self.assertEqual(self.guidance.program_number, 65)
                self.assertEqual(self.guidance.wch_vert, -2)
                self.assertTrue(self.guidance.x_override)
                self.assertEqual(self.guidance.wch_phase, 2)

    def test_integration_p65_start_and_guidance(self):
        """Test integration of P65 start and guidance execution"""
        # Initialize P65
        self.guidance.p65_start()

        # Set realistic lunar landing conditions
        position = np.array([2.0, 1.0, -30.0])  # 30m altitude, slight horizontal offset
        velocity = np.array([0.5, 0.2, -1.0])   # Small horizontal drift, 1 m/s descent

        self.guidance.update_state(position, velocity)
        accel_command = self.guidance.p65_guidance()

        # Verify system is in correct state
        self.assertEqual(self.guidance.program_number, 65)
        self.assertEqual(self.guidance.wch_phase, 2)

        # Verify guidance produces reasonable output
        self.assertIsInstance(accel_command, np.ndarray)
        self.assertEqual(len(accel_command), 3)

        # Verify display variables are reasonable
        self.assertEqual(self.guidance.altitude, 30.0)
        self.assertEqual(self.guidance.alt_rate, 1.0)
        self.assertAlmostEqual(self.guidance.hor_velocity, np.sqrt(0.29), places=5)


if __name__ == '__main__':
    unittest.main()
