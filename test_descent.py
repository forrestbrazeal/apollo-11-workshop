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

        # Calculate expected acceleration using correct AGC P65 equation
        # AGC P65: ACG = (V2FG - VGU) / TAUVERT - G_EFF
        velocity_error = self.guidance.v2fg - self.guidance.v_gu
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

if __name__ == '__main__':
    unittest.main()