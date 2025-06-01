const { LunarDescentGuidance } = require('./descent.js');

function validateSimulationResults(actualTime, actualVelocity, actualAltitude, actualHorVel, actualVertVel, actualPosition) {
  /**
   * Validate simulation results against expected parameters and provide detailed feedback.
   *
   * Expected results:
   * - TOUCHDOWN at time: 7.1 seconds (exact)
   * - Final X velocity: between 0 and 1 m/s
   * - Final Y velocity: should be less than 0 m/s
   * - Final Z velocity: should be less than 20 and greater than 0 m/s
   * - Final altitude: between -1 and 1 m
   * - Horizontal velocity at touchdown: between 0 and 1 m/s
   * - Vertical velocity at touchdown: less than 0 m/s
   */
  console.log("\n" + "=".repeat(60));
  console.log("SIMULATION RESULTS VALIDATION");
  console.log("=".repeat(60));

  // Display final position (without validation)
  console.log(`📍 FINAL POSITION: [${actualPosition[0].toFixed(1)}, ${actualPosition[1].toFixed(1)}, ${actualPosition[2].toFixed(1)}] m`);

  // Expected values and ranges
  const expectedTime = 7.1;
  // Velocity ranges: [min_x, max_x], [min_y, max_y], [min_z, max_z]
  const velocityRanges = [[0.0, 1.0], [-Infinity, 0.0], [0.0, 20.0]];
  const altitudeRange = [-1.0, 1.0];
  const horVelRange = [0.0, 1.0];
  const vertVelMax = 0.0;  // Should be less than 0

  // Tolerance for floating point comparisons
  const timeTolerance = 0.1;

  let validationPassed = true;

  // Check touchdown time (exact validation)
  const timeDiff = Math.abs(actualTime - expectedTime);
  if (timeDiff <= timeTolerance) {
    console.log(`✅ TOUCHDOWN TIME: ${actualTime.toFixed(1)}s (Expected: ${expectedTime.toFixed(1)}s) - CORRECT`);
  } else {
    console.log(`❌ TOUCHDOWN TIME: ${actualTime.toFixed(1)}s (Expected: ${expectedTime.toFixed(1)}s)`);
    if (actualTime > expectedTime) {
      console.log(`   → The landing took too long! (${timeDiff.toFixed(1)}s slower than expected)`);
    } else {
      console.log(`   → The landing was too fast! (${timeDiff.toFixed(1)}s faster than expected)`);
    }
    validationPassed = false;
  }

  // Check final velocity components (range validation)
  const velocityErrors = [];
  const componentNames = ['X', 'Y', 'Z'];
  const rangeDescriptions = ['between 0 and 1', 'less than 0', 'between 0 and 20'];

  for (let i = 0; i < 3; i++) {
    const actual = actualVelocity[i];
    const [minVal, maxVal] = velocityRanges[i];
    const componentName = componentNames[i];
    const rangeDesc = rangeDescriptions[i];

    if (minVal <= actual && actual <= maxVal) {
      console.log(`✅ FINAL VELOCITY ${componentName}: ${actual.toFixed(2)} m/s (${rangeDesc} m/s) - CORRECT`);
    } else {
      console.log(`❌ FINAL VELOCITY ${componentName}: ${actual.toFixed(2)} m/s (should be ${rangeDesc} m/s)`);
      if (actual < minVal) {
        console.log(`   → The ${componentName.toLowerCase()}-velocity is too low! (${actual.toFixed(2)} < ${minVal.toFixed(2)})`);
      } else {
        console.log(`   → The ${componentName.toLowerCase()}-velocity is too high! (${actual.toFixed(2)} > ${maxVal.toFixed(2)})`);
      }
      velocityErrors.push(componentName);
      validationPassed = false;
    }
  }

  // Check final altitude (range validation)
  const [minAlt, maxAlt] = altitudeRange;
  if (minAlt <= actualAltitude && actualAltitude <= maxAlt) {
    console.log(`✅ FINAL ALTITUDE: ${actualAltitude.toFixed(2)}m (between ${minAlt.toFixed(1)} and ${maxAlt.toFixed(1)}m) - CORRECT`);
  } else {
    console.log(`❌ FINAL ALTITUDE: ${actualAltitude.toFixed(2)}m (should be between ${minAlt.toFixed(1)} and ${maxAlt.toFixed(1)}m)`);
    if (actualAltitude < minAlt) {
      console.log(`   → The final altitude is too low! (${actualAltitude.toFixed(2)}m < ${minAlt.toFixed(1)}m)`);
    } else {
      console.log(`   → The final altitude is too high! (${actualAltitude.toFixed(2)}m > ${maxAlt.toFixed(1)}m)`);
    }
    validationPassed = false;
  }

  // Check horizontal velocity at touchdown (range validation)
  const [minHorVel, maxHorVel] = horVelRange;
  if (minHorVel <= actualHorVel && actualHorVel <= maxHorVel) {
    console.log(`✅ HORIZONTAL VELOCITY: ${actualHorVel.toFixed(2)} m/s (between ${minHorVel.toFixed(1)} and ${maxHorVel.toFixed(1)} m/s) - CORRECT`);
  } else {
    console.log(`❌ HORIZONTAL VELOCITY: ${actualHorVel.toFixed(2)} m/s (should be between ${minHorVel.toFixed(1)} and ${maxHorVel.toFixed(1)} m/s)`);
    if (actualHorVel < minHorVel) {
      console.log(`   → The horizontal velocity is too low! (${actualHorVel.toFixed(2)} < ${minHorVel.toFixed(1)})`);
    } else {
      console.log(`   → The horizontal velocity is too high! (${actualHorVel.toFixed(2)} > ${maxHorVel.toFixed(1)})`);
    }
    validationPassed = false;
  }

  // Check vertical velocity at touchdown (range validation)
  if (actualVertVel < vertVelMax) {
    console.log(`✅ VERTICAL VELOCITY: ${actualVertVel.toFixed(2)} m/s (less than ${vertVelMax.toFixed(1)} m/s) - CORRECT`);
  } else {
    console.log(`❌ VERTICAL VELOCITY: ${actualVertVel.toFixed(2)} m/s (should be less than ${vertVelMax.toFixed(1)} m/s)`);
    console.log(`   → The vertical velocity should be negative (descending)! (${actualVertVel.toFixed(2)} >= ${vertVelMax.toFixed(1)})`);
    validationPassed = false;
  }

  // Overall validation result
  console.log("-".repeat(60));
  if (validationPassed) {
    console.log("🎯 OVERALL VALIDATION: ALL PARAMETERS CORRECT!");
    console.log("   The simulation is producing the expected results.");
  } else {
    console.log("⚠️  OVERALL VALIDATION: SOME PARAMETERS ARE INCORRECT!");
    console.log("   The simulation results deviate from expected values.");
    if (velocityErrors.length > 0) {
      console.log(`   Velocity components with errors: ${velocityErrors.join(', ')}`);
    }
  }

  console.log("=".repeat(60));
}

// For browser compatibility
if (typeof window !== 'undefined') {
  // Use a plotting library like Chart.js in browser environments
  // This is a placeholder - actual implementation would depend on the plotting library
  plotData = (data, options) => {
    console.log("Plotting would happen here in browser environment");
    console.log("Data:", data);
  };
} else {
  // Node.js environment - we'd need a different plotting solution
  // For simplicity, we'll just log that plotting would happen
  plotData = (data, options) => {
    console.log("Plotting would happen here in Node.js environment");
    console.log("Data:", data);
  };
}

async function run_p65_simulation() {
  /**
   * Example usage of the P65 guidance algorithm with visualization
   */
  // Create the guidance instance
  const lm_guidance = new LunarDescentGuidance();

  // Initialize P65
  lm_guidance.p65_start();

  // Simulation parameters
  const dt = 0.1;  // Time step (seconds)
  const max_time = 45.0;  // Maximum simulation time (seconds)

  // Initial conditions
  // Starting at 150 feet (45.72m) altitude with some horizontal offset and velocity
  let position = [20.0, -15.0, -45.72];  // [x, y, z] in meters
  let velocity = [2.0, -1.5, -0.3];      // [vx, vy, vz] in m/s

  // Engine parameters (simplified)
  const max_thrust = 4500.0;  // Maximum thrust in Newtons
  const min_thrust = 1200.0;  // Minimum thrust in Newtons
  const lm_mass = 7500.0;     // LM mass in kg (simplified, would decrease as fuel is used)

  // Storage for plotting
  const time_points = [];
  const altitude_points = [];
  const horiz_vel_points = [];
  const vert_vel_points = [];
  const x_points = [];
  const y_points = [];
  const z_points = [];
  const thrust_points = [];

  // Main simulation loop
  let time = 0.0;
  console.log("Starting P65 Vertical Descent simulation...");
  let touchdown = false;

  while (time < max_time) {
    // Update guidance with current state
    lm_guidance.update_state(position, velocity);

    // Run the P65 guidance algorithm
    const accel_command = lm_guidance.p65_guidance();

    // Check for touchdown (altitude less than 0.5m)
    const altitude = -position[2];  // Convert to positive altitude
    if (altitude < 0.5) {  // Within 0.5m of surface
      console.log(`\nTOUCHDOWN at time ${time.toFixed(1)} seconds!`);
      console.log(`Final position: [${position[0].toFixed(1)}, ${position[1].toFixed(1)}, ${position[2].toFixed(1)}] m`);
      console.log(`Final velocity: [${velocity[0].toFixed(2)}, ${velocity[1].toFixed(2)}, ${velocity[2].toFixed(2)}] m/s`);
      console.log(`Final altitude: ${altitude.toFixed(2)} m`);
      console.log(`Horizontal velocity at touchdown: ${lm_guidance.hor_velocity.toFixed(2)} m/s`);
      console.log(`Vertical velocity at touchdown: ${(-velocity[2]).toFixed(2)} m/s`);

      // Validate simulation results against expected parameters
      validateSimulationResults(time, velocity, altitude, lm_guidance.hor_velocity, -velocity[2], position);

      touchdown = true;
      break;
    }

    // Convert acceleration command to thrust (simplified)
    // In reality, this would involve the RCS jets for horizontal control
    // and the descent engine for vertical control
    let thrust_magnitude = 0;
    let actual_accel = [0, 0, 0];

    const accel_norm = Math.sqrt(
      accel_command[0]**2 + accel_command[1]**2 + accel_command[2]**2
    );

    if (accel_norm > 0) {
      thrust_magnitude = lm_mass * accel_norm;

      // Apply thrust limits
      thrust_magnitude = Math.min(max_thrust, Math.max(min_thrust, thrust_magnitude));

      // Calculate actual acceleration (including gravity)
      const thrust_direction = [
        accel_command[0] / accel_norm,
        accel_command[1] / accel_norm,
        accel_command[2] / accel_norm
      ];

      actual_accel = [
        (thrust_magnitude / lm_mass) * thrust_direction[0] - lm_guidance.lunar_gravity[0],
        (thrust_magnitude / lm_mass) * thrust_direction[1] - lm_guidance.lunar_gravity[1],
        (thrust_magnitude / lm_mass) * thrust_direction[2] - lm_guidance.lunar_gravity[2]
      ];
    } else {
      // No thrust command
      thrust_magnitude = 0;
      actual_accel = [...lm_guidance.lunar_gravity].map(v => -v);
    }

    // Update state with simple integration
    velocity = [
      velocity[0] + actual_accel[0] * dt,
      velocity[1] + actual_accel[1] * dt,
      velocity[2] + actual_accel[2] * dt
    ];

    position = [
      position[0] + velocity[0] * dt,
      position[1] + velocity[1] * dt,
      position[2] + velocity[2] * dt
    ];

    // Store data for plotting
    time_points.push(time);
    altitude_points.push(-position[2]);  // Convert to positive altitude
    horiz_vel_points.push(lm_guidance.hor_velocity);
    vert_vel_points.push(-velocity[2]);  // Convert to positive descent rate
    x_points.push(position[0]);
    y_points.push(position[1]);
    z_points.push(position[2]);
    thrust_points.push(thrust_magnitude);

    // Display status every 5 seconds
    if (Math.round(time * 10) % 50 === 0) {
      lm_guidance.display_status();
    }

    time += dt;
  }

  // Print final status if simulation ended without touchdown
  if (!touchdown) {
    lm_guidance.update_state(position, velocity);
    lm_guidance.p65_guidance();
    console.log(`\nSimulation ended at time ${time.toFixed(1)} seconds (time limit reached)`);
    console.log(`Final position: [${position[0].toFixed(1)}, ${position[1].toFixed(1)}, ${position[2].toFixed(1)}] m`);
    console.log(`Final velocity: [${velocity[0].toFixed(2)}, ${velocity[1].toFixed(2)}, ${velocity[2].toFixed(2)}] m/s`);
    console.log(`Final altitude: ${(-position[2]).toFixed(2)} m`);
    console.log(`Final horizontal velocity: ${lm_guidance.hor_velocity.toFixed(2)} m/s`);
    console.log(`Final vertical velocity: ${(-velocity[2]).toFixed(2)} m/s`);

    // Validation for failed landing
    console.log("\n" + "=".repeat(60));
    console.log("SIMULATION RESULTS VALIDATION");
    console.log("=".repeat(60));
    console.log(`📍 FINAL POSITION: [${position[0].toFixed(1)}, ${position[1].toFixed(1)}, ${position[2].toFixed(1)}] m`);
    console.log("❌ TOUCHDOWN FAILURE: Simulation ended without successful landing!");
    console.log(`   → Expected touchdown at 7.1 seconds, but simulation ran to ${time.toFixed(1)} seconds`);
    console.log(`   → Final altitude was ${(-position[2]).toFixed(2)}m (should be between -1 and 1m)`);
    console.log("⚠️  OVERALL VALIDATION: LANDING FAILED!");
    console.log("   The guidance system did not achieve the expected landing.");
    console.log("=".repeat(60));
  }

  // Create visualization data
  const plotData = {
    time: time_points,
    altitude: altitude_points,
    horizVel: horiz_vel_points,
    vertVel: vert_vel_points,
    x: x_points,
    y: y_points,
    z: z_points,
    thrust: thrust_points
  };

  // Return the data for potential further processing
  return plotData;
}

// Export for use in other modules
if (typeof module !== 'undefined') {
  module.exports = { run_p65_simulation };
}

// Run the simulation directly when this file is executed
if (typeof require !== 'undefined' && require.main === module) {
  run_p65_simulation();
}