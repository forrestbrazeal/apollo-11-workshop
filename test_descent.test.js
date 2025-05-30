/**
 * Test cases for the LunarDescentGuidance class
 * JavaScript equivalent of test_descent.py
 */

const { LunarDescentGuidance } = require('./descent.js');

describe('LunarDescentGuidance', () => {
  let guidance;

  beforeEach(() => {
    guidance = new LunarDescentGuidance();
  });

  describe('initialization', () => {
    test('should initialize with correct default values', () => {
      expect(guidance.dt).toBe(0.1);
      expect(guidance.tau_vert).toBe(5.0);
      expect(guidance.program_number).toBe(0);
      expect(guidance.wch_phase).toBe(0);
      expect(guidance.wch_vert).toBe(0);
      expect(guidance.x_override).toBe(false);

      // Check arrays
      expect(guidance.lunar_gravity).toEqual([0.0, 0.0, -1.622]);
      expect(guidance.r_gu).toEqual([0.0, 0.0, 0.0]);
      expect(guidance.v_gu).toEqual([0.0, 0.0, 0.0]);
      expect(guidance.v2fg).toEqual([0.0, 0.0, -1.22]);
      expect(guidance.g_eff).toEqual([0.0, 0.0, -1.622]);
      expect(guidance.a_command).toEqual([0.0, 0.0, 0.0]);
    });

    test('should initialize display variables correctly', () => {
      expect(guidance.altitude).toBe(0.0);
      expect(guidance.alt_rate).toBe(0.0);
      expect(guidance.hor_velocity).toBe(0.0);
    });
  });

  describe('p65_start', () => {
    test('should set correct values when P65 starts', () => {
      // Capture console output
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      guidance.p65_start();

      expect(guidance.program_number).toBe(65);
      expect(guidance.wch_vert).toBe(-2);
      expect(guidance.x_override).toBe(true);
      expect(guidance.wch_phase).toBe(2);
      
      expect(consoleSpy).toHaveBeenCalledWith("P65 Vertical Descent Guidance Started");
      consoleSpy.mockRestore();
    });
  });

  describe('update_state', () => {
    test('should update state with position and velocity vectors', () => {
      const test_position = [10.0, 20.0, -30.0];
      const test_velocity = [1.0, 2.0, -3.0];

      guidance.update_state(test_position, test_velocity);

      expect(guidance.r_gu).toEqual(test_position);
      expect(guidance.v_gu).toEqual(test_velocity);
    });
  });

  describe('horizontal velocity calculation', () => {
    const testCases = [
      { velocity: [3.0, 4.0, -1.0], expected: 5.0 },  // 3-4-5 triangle
      { velocity: [0.0, 0.0, -2.0], expected: 0.0 },  // No horizontal velocity
      { velocity: [1.0, 1.0, -1.0], expected: Math.sqrt(2.0) },  // Equal X and Y components
      { velocity: [-2.0, 2.0, -1.0], expected: Math.sqrt(8.0) }   // Negative X component
    ];

    testCases.forEach(({ velocity, expected }) => {
      test(`should calculate horizontal velocity correctly for velocity ${JSON.stringify(velocity)}`, () => {
        guidance.v_gu = velocity;
        guidance.p65_guidance();
        expect(guidance.hor_velocity).toBeCloseTo(expected, 5);
      });
    });
  });

  describe('gravity effect on guidance', () => {
    test('should properly account for gravity in guidance', () => {
      // Set zero velocity error (current = desired)
      guidance.v_gu = [0.0, 0.0, -1.22];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accel_command = guidance.p65_guidance();

      // Should command upward acceleration to counteract downward gravity
      const expected_accel = [0.0, 0.0, 1.622];
      expect(accel_command[0]).toBeCloseTo(expected_accel[0], 6);
      expect(accel_command[1]).toBeCloseTo(expected_accel[1], 6);
      expect(accel_command[2]).toBeCloseTo(expected_accel[2], 6);
    });
  });

  describe('p65_guidance updates display variables', () => {
    test('should update altitude, alt_rate, and horizontal velocity', () => {
      // Set test position and velocity
      const test_position = [3.0, 4.0, -50.0];  // 50m altitude
      const test_velocity = [1.0, 2.0, -1.5];   // 1.5 m/s descent, sqrt(5) horizontal

      guidance.update_state(test_position, test_velocity);
      guidance.p65_guidance();

      expect(guidance.altitude).toBeCloseTo(50.0);
      expect(guidance.alt_rate).toBeCloseTo(1.5);
      expect(guidance.hor_velocity).toBeCloseTo(Math.sqrt(5.0), 5);
    });
  });

  describe('altitude calculation with positive Z', () => {
    test('should calculate negative altitude for positive Z position', () => {
      guidance.r_gu = [0.0, 0.0, 100.0];
      guidance.p65_guidance();
      expect(guidance.altitude).toBe(-100.0);  // Negative altitude (below surface)
    });
  });
});

// Helper function to run tests if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
  console.log('Running tests...');
  // Note: In a real environment, you would use: npm test or jest
  console.log('Please run: npm test or jest test_descent.test.js');
}
