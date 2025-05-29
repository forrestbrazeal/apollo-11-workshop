const { LunarDescentGuidance } = require('../Apollo-11/descent.js');

describe('LunarDescentGuidance', () => {
  let guidance;

  beforeEach(() => {
    guidance = new LunarDescentGuidance();
  });

  describe('Initialization', () => {
    test('should initialize with correct default values', () => {
      expect(guidance.dt).toBe(0.1);
      expect(guidance.tau_vert).toBe(5.0);
      expect(guidance.program_number).toBe(0);
      expect(guidance.wch_phase).toBe(0);
      expect(guidance.wch_vert).toBe(0);
      expect(guidance.x_override).toBe(false);

      // Check arrays
      expect(guidance.lunar_gravity).toEqual([0.0, 0.0, -1.622]);
      expect(guidance.r_gu).toEqual([0, 0, 0]);
      expect(guidance.v_gu).toEqual([0, 0, 0]);
      expect(guidance.v2fg).toEqual([0.0, 0.0, -1.22]);
      expect(guidance.g_eff).toEqual([0.0, 0.0, -1.622]);
      expect(guidance.a_command).toEqual([0, 0, 0]);
    });

    test('should initialize display variables correctly', () => {
      expect(guidance.altitude).toBe(0.0);
      expect(guidance.alt_rate).toBe(0.0);
      expect(guidance.hor_velocity).toBe(0.0);
    });
  });

  describe('P65 Start Sequence', () => {
    test('should set correct values when P65 starts', () => {
      // Mock console.log to avoid output during tests
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      guidance.p65_start();

      expect(guidance.program_number).toBe(65);
      expect(guidance.wch_vert).toBe(-2);
      expect(guidance.x_override).toBe(true);
      expect(guidance.wch_phase).toBe(2);
      expect(consoleSpy).toHaveBeenCalledWith("P65 INITIALIZED: Automatic Vertical Descent");

      consoleSpy.mockRestore();
    });

    test('should maintain correct state across multiple P65 start calls', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      for (let i = 0; i < 3; i++) {
        guidance.p65_start();

        expect(guidance.program_number).toBe(65);
        expect(guidance.wch_vert).toBe(-2);
        expect(guidance.x_override).toBe(true);
        expect(guidance.wch_phase).toBe(2);
      }

      consoleSpy.mockRestore();
    });
  });

  describe('State Updates', () => {
    test('should update state with position and velocity arrays', () => {
      const testPosition = [10.0, 20.0, -30.0];
      const testVelocity = [1.0, 2.0, -3.0];

      guidance.update_state(testPosition, testVelocity);

      expect(guidance.r_gu).toEqual(testPosition);
      expect(guidance.v_gu).toEqual(testVelocity);
    });

    test('should create independent copies of input arrays', () => {
      const testPosition = [5.0, -10.0, 15.0];
      const testVelocity = [-2.0, 3.0, 1.0];

      guidance.update_state(testPosition, testVelocity);

      // Modify original arrays
      testPosition[0] = 999;
      testVelocity[0] = 999;

      // Guidance state should be unchanged
      expect(guidance.r_gu[0]).toBe(5.0);
      expect(guidance.v_gu[0]).toBe(-2.0);
    });

    test('should persist state correctly across multiple function calls', () => {
      const initialPos = [5.0, 10.0, -20.0];
      const initialVel = [0.5, -0.3, -1.5];

      guidance.update_state(initialPos, initialVel);
      guidance.p65_guidance();

      // Verify state is preserved
      expect(guidance.r_gu).toEqual(initialPos);
      expect(guidance.v_gu).toEqual(initialVel);

      // Update state again
      const newPos = [5.1, 9.9, -19.8];
      const newVel = [0.4, -0.2, -1.4];

      guidance.update_state(newPos, newVel);

      // Verify new state is stored
      expect(guidance.r_gu).toEqual(newPos);
      expect(guidance.v_gu).toEqual(newVel);
    });
  });

  describe('Descent Rate Setting', () => {
    test('should set positive descent rate correctly', () => {
      guidance.set_descent_rate(2.0);
      expect(guidance.v2fg[2]).toBe(-2.0);
    });

    test('should set zero descent rate for hover', () => {
      guidance.set_descent_rate(0.0);
      expect(guidance.v2fg[2]).toBeCloseTo(0.0, 10);
    });

    test('should set negative descent rate for ascent', () => {
      guidance.set_descent_rate(-1.5);
      expect(guidance.v2fg[2]).toBe(1.5);
    });

    test('should handle boundary values for descent rate', () => {
      const testRates = [0.001, 0.1, 1.0, 5.0, 10.0, 100.0];

      testRates.forEach(rate => {
        guidance.set_descent_rate(rate);
        expect(guidance.v2fg[2]).toBe(-rate);
      });
    });
  });

  describe('P65 Guidance Algorithm', () => {
    test('should command correct acceleration with zero velocity error', () => {
      // Set current velocity to match desired velocity
      guidance.v_gu = [0.0, 0.0, -1.22];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accelCommand = guidance.p65_guidance();

      // Should command acceleration to counteract gravity only
      const expectedAccel = [0.0, 0.0, 1.622];
      expect(accelCommand).toEqual(expectedAccel);
    });

    test('should handle velocity error correctly', () => {
      // Current velocity different from desired
      guidance.v_gu = [1.0, 0.5, -2.0];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accelCommand = guidance.p65_guidance();

      // Calculate expected acceleration manually
      const velocityError = [1.0, 0.5, -0.78];
      const expectedAccel = [
        velocityError[0] / 5.0 - 0.0,
        velocityError[1] / 5.0 - 0.0,
        velocityError[2] / 5.0 - (-1.622)
      ];

      expect(accelCommand[0]).toBeCloseTo(expectedAccel[0], 5);
      expect(accelCommand[1]).toBeCloseTo(expectedAccel[1], 5);
      expect(accelCommand[2]).toBeCloseTo(expectedAccel[2], 5);
    });

    test('should update display variables correctly', () => {
      const testPosition = [3.0, 4.0, -50.0];  // 50m altitude
      const testVelocity = [1.0, 2.0, -1.5];   // 1.5 m/s descent, sqrt(5) horizontal

      guidance.update_state(testPosition, testVelocity);
      guidance.p65_guidance();

      expect(guidance.altitude).toBe(50.0);
      expect(guidance.alt_rate).toBe(1.5);
      expect(guidance.hor_velocity).toBeCloseTo(Math.sqrt(5.0), 5);
    });

    test('should store acceleration command correctly', () => {
      guidance.v_gu = [1.0, 2.0, -3.0];

      const accelCommand = guidance.p65_guidance();

      // Verify the returned command matches stored command
      expect(accelCommand).toEqual(guidance.a_command);
    });

    test('should handle extreme velocity errors', () => {
      // Set very large velocity error
      guidance.v_gu = [10.0, -5.0, 5.0];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accelCommand = guidance.p65_guidance();

      // Verify the acceleration command is calculated correctly
      const velocityError = [10.0, -5.0, 6.22];
      const expectedAccel = [
        velocityError[0] / 5.0 - 0.0,
        velocityError[1] / 5.0 - 0.0,
        velocityError[2] / 5.0 - (-1.622)
      ];

      expect(accelCommand[0]).toBeCloseTo(expectedAccel[0], 5);
      expect(accelCommand[1]).toBeCloseTo(expectedAccel[1], 5);
      expect(accelCommand[2]).toBeCloseTo(expectedAccel[2], 5);
    });

    test('should work with different time constants', () => {
      guidance.tau_vert = 10.0;  // Slower response
      guidance.v_gu = [2.0, 0.0, -2.0];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accelCommand = guidance.p65_guidance();

      const velocityError = [2.0, 0.0, -0.78];
      const expectedAccel = [
        velocityError[0] / 10.0 - 0.0,
        velocityError[1] / 10.0 - 0.0,
        velocityError[2] / 10.0 - (-1.622)
      ];

      expect(accelCommand[0]).toBeCloseTo(expectedAccel[0], 5);
      expect(accelCommand[1]).toBeCloseTo(expectedAccel[1], 5);
      expect(accelCommand[2]).toBeCloseTo(expectedAccel[2], 5);
    });
  });

  describe('Display Variable Calculations', () => {
    test('should calculate altitude with positive Z position', () => {
      guidance.r_gu = [0.0, 0.0, 100.0];
      guidance.p65_guidance();
      expect(guidance.altitude).toBe(-100.0);  // Negative altitude (below surface)
    });

    test('should calculate altitude with negative Z position', () => {
      guidance.r_gu = [0.0, 0.0, -75.0];
      guidance.p65_guidance();
      expect(guidance.altitude).toBe(75.0);  // Positive altitude (above surface)
    });

    test('should calculate horizontal velocity magnitude correctly', () => {
      const testCases = [
        { velocity: [3.0, 4.0, -1.0], expected: 5.0 },  // 3-4-5 triangle
        { velocity: [0.0, 0.0, -2.0], expected: 0.0 },  // No horizontal velocity
        { velocity: [1.0, 1.0, -1.0], expected: Math.sqrt(2.0) },  // Equal X and Y
        { velocity: [-2.0, 2.0, -1.0], expected: Math.sqrt(8.0) }   // Negative X
      ];

      testCases.forEach(({ velocity, expected }) => {
        guidance.v_gu = velocity;
        guidance.p65_guidance();
        expect(guidance.hor_velocity).toBeCloseTo(expected, 5);
      });
    });

    test('should handle zero values correctly', () => {
      guidance.update_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
      guidance.p65_guidance();

      expect(guidance.altitude).toBeCloseTo(0.0, 10);
      expect(guidance.alt_rate).toBeCloseTo(0.0, 10);
      expect(guidance.hor_velocity).toBeCloseTo(0.0, 10);
    });
  });

  describe('Mode Switching', () => {
    test('should switch to P67 when manual throttle is active', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      guidance.program_number = 65;

      const result = guidance.check_mode_switch(true, false);

      expect(result).toBe(67);
      expect(consoleSpy).toHaveBeenCalledWith("SWITCHING TO P67: Manual Control Mode");

      consoleSpy.mockRestore();
    });

    test('should switch to P66 when ROD switch is clicked', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      guidance.program_number = 65;

      const result = guidance.check_mode_switch(false, true);

      expect(result).toBe(66);
      expect(consoleSpy).toHaveBeenCalledWith("SWITCHING TO P66: Rate of Descent Mode");

      consoleSpy.mockRestore();
    });

    test('should prioritize manual throttle when both conditions are true', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      guidance.program_number = 65;

      // Manual throttle should take priority over ROD switch
      const result = guidance.check_mode_switch(true, true);

      expect(result).toBe(67);
      expect(consoleSpy).toHaveBeenCalledWith("SWITCHING TO P67: Manual Control Mode");

      consoleSpy.mockRestore();
    });

    test('should not switch when no conditions are met', () => {
      guidance.program_number = 65;

      const result = guidance.check_mode_switch(false, false);

      expect(result).toBe(65);
    });
  });

  describe('Vector Utility Functions', () => {
    test('should subtract vectors correctly', () => {
      const a = [5.0, 3.0, -2.0];
      const b = [1.0, 2.0, -1.0];

      const result = guidance.subtractVectors(a, b);

      expect(result).toEqual([4.0, 1.0, -1.0]);
    });

    test('should scale vectors correctly', () => {
      const v = [2.0, -4.0, 6.0];
      const scalar = 0.5;

      const result = guidance.scaleVector(v, scalar);

      expect(result).toEqual([1.0, -2.0, 3.0]);
    });

    test('should handle zero vectors in subtraction', () => {
      const a = [0.0, 0.0, 0.0];
      const b = [1.0, 2.0, 3.0];

      const result = guidance.subtractVectors(a, b);

      expect(result).toEqual([-1.0, -2.0, -3.0]);
    });

    test('should handle zero scalar in scaling', () => {
      const v = [5.0, -3.0, 2.0];
      const scalar = 0.0;

      const result = guidance.scaleVector(v, scalar);

      expect(result[0]).toBeCloseTo(0.0, 10);
      expect(result[1]).toBeCloseTo(0.0, 10);
      expect(result[2]).toBeCloseTo(0.0, 10);
    });

    test('should handle negative scalar in scaling', () => {
      const v = [1.0, -2.0, 3.0];
      const scalar = -2.0;

      const result = guidance.scaleVector(v, scalar);

      expect(result).toEqual([-2.0, 4.0, -6.0]);
    });
  });

  describe('Multiple Guidance Cycles', () => {
    test('should handle consecutive guidance cycles correctly', () => {
      const positions = [
        [0.0, 0.0, -100.0],
        [1.0, 0.5, -95.0],
        [2.0, 1.0, -90.0]
      ];
      const velocities = [
        [1.0, 0.5, -1.0],
        [1.0, 0.5, -1.1],
        [1.0, 0.5, -1.2]
      ];

      positions.forEach((pos, i) => {
        const vel = velocities[i];
        guidance.update_state(pos, vel);
        const accelCommand = guidance.p65_guidance();

        // Verify acceleration command is computed
        expect(Array.isArray(accelCommand)).toBe(true);
        expect(accelCommand.length).toBe(3);

        // Verify display variables are updated
        expect(guidance.altitude).toBe(-pos[2]);
        expect(guidance.alt_rate).toBe(-vel[2]);
      });
    });
  });

  describe('Gravity Effects', () => {
    test('should properly account for gravity in guidance', () => {
      // Set zero velocity error (current = desired)
      guidance.v_gu = [0.0, 0.0, -1.22];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accelCommand = guidance.p65_guidance();

      // Should command upward acceleration to counteract downward gravity
      const expectedAccel = [0.0, 0.0, 1.622];
      expect(accelCommand[0]).toBeCloseTo(expectedAccel[0], 5);
      expect(accelCommand[1]).toBeCloseTo(expectedAccel[1], 5);
      expect(accelCommand[2]).toBeCloseTo(expectedAccel[2], 5);
    });
  });

  describe('Integration Tests', () => {
    test('should integrate P65 start and guidance execution correctly', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      // Initialize P65
      guidance.p65_start();

      // Set realistic lunar landing conditions
      const position = [2.0, 1.0, -30.0];  // 30m altitude, slight horizontal offset
      const velocity = [0.5, 0.2, -1.0];   // Small horizontal drift, 1 m/s descent

      guidance.update_state(position, velocity);
      const accelCommand = guidance.p65_guidance();

      // Verify system is in correct state
      expect(guidance.program_number).toBe(65);
      expect(guidance.wch_phase).toBe(2);

      // Verify guidance produces reasonable output
      expect(Array.isArray(accelCommand)).toBe(true);
      expect(accelCommand.length).toBe(3);

      // Verify display variables are reasonable
      expect(guidance.altitude).toBe(30.0);
      expect(guidance.alt_rate).toBe(1.0);
      expect(guidance.hor_velocity).toBeCloseTo(Math.sqrt(0.29), 5);

      consoleSpy.mockRestore();
    });

    test('should maintain consistency across multiple operations', () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      // Initialize and run multiple cycles
      guidance.p65_start();

      for (let i = 0; i < 5; i++) {
        const pos = [i * 0.1, i * 0.05, -50 + i];
        const vel = [0.1, 0.05, -1.0 - i * 0.1];

        guidance.update_state(pos, vel);
        const accel = guidance.p65_guidance();

        expect(guidance.program_number).toBe(65);
        expect(Array.isArray(accel)).toBe(true);
        expect(accel.length).toBe(3);
      }

      consoleSpy.mockRestore();
    });
  });

  describe('Edge Cases and Robustness', () => {
    test('should handle very small numbers correctly', () => {
      guidance.v_gu = [1e-10, 1e-10, -1.22];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accelCommand = guidance.p65_guidance();

      expect(accelCommand[0]).toBeCloseTo(1e-10 / 5.0, 15);
      expect(accelCommand[1]).toBeCloseTo(1e-10 / 5.0, 15);
      expect(accelCommand[2]).toBeCloseTo(1.622, 5);
    });

    test('should handle large numbers correctly', () => {
      guidance.v_gu = [1000.0, -500.0, 100.0];
      guidance.v2fg = [0.0, 0.0, -1.22];

      const accelCommand = guidance.p65_guidance();

      expect(accelCommand[0]).toBeCloseTo(1000.0 / 5.0, 5);
      expect(accelCommand[1]).toBeCloseTo(-500.0 / 5.0, 5);
      expect(accelCommand[2]).toBeCloseTo((101.22) / 5.0 + 1.622, 5);
    });
  });
});
