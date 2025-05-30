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

    def test_gravity_effect_on_guidance(self):
        """Test that gravity is properly accounted for in guidance"""
        # Set zero velocity error (current = desired)
        self.guidance.v_gu = np.array([0.0, 0.0, -1.22])
        self.guidance.v2fg = np.array([0.0, 0.0, -1.22])

        accel_command = self.guidance.p65_guidance()

        # Should command upward acceleration to counteract downward gravity
        expected_accel = np.array([0.0, 0.0, 1.622])
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


if __name__ == '__main__':
    unittest.main()