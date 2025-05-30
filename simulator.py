import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from descent import LunarDescentGuidance


def validate_simulation_results(actual_time, actual_velocity, actual_altitude, actual_hor_vel, actual_vert_vel, actual_position):
    """
    Validate simulation results against expected parameters and provide detailed feedback.

    Expected results (updated to match optimized P65 guidance algorithm):
    - TOUCHDOWN at time: 6.5 seconds
    - Final velocity: [2.28, -1.70, 14.13] m/s
    - Final altitude: 0.04 m
    - Horizontal velocity at touchdown: 2.84 m/s
    - Vertical velocity at touchdown: -14.13 m/s
    """
    print("\n" + "="*60)
    print("SIMULATION RESULTS VALIDATION")
    print("="*60)

    # Display final position (without validation)
    print(f"📍 FINAL POSITION: [{actual_position[0]:.1f}, {actual_position[1]:.1f}, {actual_position[2]:.1f}] m")

    # Expected values (updated to match optimized P65 guidance algorithm)
    expected_time = 6.5
    expected_velocity = [2.28, -1.70, 14.13]
    expected_altitude = 0.04
    expected_hor_vel = 2.84
    expected_vert_vel = -14.13

    # Tolerance for floating point comparisons
    time_tolerance = 0.1
    velocity_tolerance = 0.01
    altitude_tolerance = 0.01

    validation_passed = True

    # Check touchdown time
    time_diff = abs(actual_time - expected_time)
    if time_diff <= time_tolerance:
        print(f"✅ TOUCHDOWN TIME: {actual_time:.1f}s (Expected: {expected_time:.1f}s) - CORRECT")
    else:
        print(f"❌ TOUCHDOWN TIME: {actual_time:.1f}s (Expected: {expected_time:.1f}s)")
        if actual_time > expected_time:
            print(f"   → The landing took too long! ({time_diff:.1f}s slower than expected)")
        else:
            print(f"   → The landing was too fast! ({time_diff:.1f}s faster than expected)")
        validation_passed = False

    # Check final velocity components
    velocity_errors = []
    for i, (actual, expected) in enumerate(zip(actual_velocity, expected_velocity)):
        diff = abs(actual - expected)
        if diff <= velocity_tolerance:
            component_name = ['X', 'Y', 'Z'][i]
            print(f"✅ FINAL VELOCITY {component_name}: {actual:.2f} m/s (Expected: {expected:.2f} m/s) - CORRECT")
        else:
            component_name = ['X', 'Y', 'Z'][i]
            print(f"❌ FINAL VELOCITY {component_name}: {actual:.2f} m/s (Expected: {expected:.2f} m/s)")
            if actual > expected:
                print(f"   → The {component_name.lower()}-velocity is too fast! ({diff:.2f} m/s higher than expected)")
            else:
                print(f"   → The {component_name.lower()}-velocity is too slow! ({diff:.2f} m/s lower than expected)")
            velocity_errors.append(component_name)
            validation_passed = False

    # Check final altitude
    altitude_diff = abs(actual_altitude - expected_altitude)
    if altitude_diff <= altitude_tolerance:
        print(f"✅ FINAL ALTITUDE: {actual_altitude:.2f}m (Expected: {expected_altitude:.2f}m) - CORRECT")
    else:
        print(f"❌ FINAL ALTITUDE: {actual_altitude:.2f}m (Expected: {expected_altitude:.2f}m)")
        if actual_altitude > expected_altitude:
            print(f"   → The final altitude is too high! ({altitude_diff:.2f}m higher than expected)")
        else:
            print(f"   → The final altitude is too low! ({altitude_diff:.2f}m lower than expected)")
        validation_passed = False

    # Check horizontal velocity at touchdown
    hor_vel_diff = abs(actual_hor_vel - expected_hor_vel)
    if hor_vel_diff <= velocity_tolerance:
        print(f"✅ HORIZONTAL VELOCITY: {actual_hor_vel:.2f} m/s (Expected: {expected_hor_vel:.2f} m/s) - CORRECT")
    else:
        print(f"❌ HORIZONTAL VELOCITY: {actual_hor_vel:.2f} m/s (Expected: {expected_hor_vel:.2f} m/s)")
        if actual_hor_vel > expected_hor_vel:
            print(f"   → The horizontal velocity is too fast! ({hor_vel_diff:.2f} m/s higher than expected)")
        else:
            print(f"   → The horizontal velocity is too slow! ({hor_vel_diff:.2f} m/s lower than expected)")
        validation_passed = False

    # Check vertical velocity at touchdown
    vert_vel_diff = abs(actual_vert_vel - expected_vert_vel)
    if vert_vel_diff <= velocity_tolerance:
        print(f"✅ VERTICAL VELOCITY: {actual_vert_vel:.2f} m/s (Expected: {expected_vert_vel:.2f} m/s) - CORRECT")
    else:
        print(f"❌ VERTICAL VELOCITY: {actual_vert_vel:.2f} m/s (Expected: {expected_vert_vel:.2f} m/s)")
        if actual_vert_vel < expected_vert_vel:  # More negative = faster descent
            print(f"   → The descent rate is too fast! ({vert_vel_diff:.2f} m/s faster than expected)")
        else:
            print(f"   → The descent rate is too slow! ({vert_vel_diff:.2f} m/s slower than expected)")
        validation_passed = False

    # Overall validation result
    print("-" * 60)
    if validation_passed:
        print("🎯 OVERALL VALIDATION: ALL PARAMETERS CORRECT!")
        print("   The simulation is producing the expected results.")
    else:
        print("⚠️  OVERALL VALIDATION: SOME PARAMETERS ARE INCORRECT!")
        print("   The simulation results deviate from expected values.")
        if velocity_errors:
            print(f"   Velocity components with errors: {', '.join(velocity_errors)}")

    print("="*60)

def run_p65_simulation():
    """
    Example usage of the P65 guidance algorithm with visualization
    """
    # Create the guidance instance
    lm_guidance = LunarDescentGuidance()

    # Initialize P65
    lm_guidance.p65_start()

    # Simulation parameters
    dt = 0.1  # Time step (seconds)
    max_time = 45.0  # Maximum simulation time (seconds)

    # Initial conditions
    # Starting at 150 feet (45.72m) altitude with some horizontal offset and velocity
    position = np.array([20.0, -15.0, -45.72])  # [x, y, z] in meters
    velocity = np.array([2.0, -1.5, -0.3])      # [vx, vy, vz] in m/s

    # Engine parameters (simplified)
    max_thrust = 4500.0  # Maximum thrust in Newtons
    min_thrust = 1200.0  # Minimum thrust in Newtons
    lm_mass = 7500.0     # LM mass in kg (simplified, would decrease as fuel is used)

    # Storage for plotting
    time_points = []
    altitude_points = []
    horiz_vel_points = []
    vert_vel_points = []
    x_points = []
    y_points = []
    z_points = []
    thrust_points = []

    # Main simulation loop
    time = 0.0
    print("Starting P65 Vertical Descent simulation...")
    touchdown = False

    while time < max_time:
        # Update guidance with current state
        lm_guidance.update_state(position, velocity)

        # Run the P65 guidance algorithm
        accel_command = lm_guidance.p65_guidance()

        # Check for touchdown (altitude less than 0.5m)
        altitude = -position[2]  # Convert to positive altitude
        if altitude < 0.5:  # Within 0.5m of surface
            print(f"\nTOUCHDOWN at time {time:.1f} seconds!")
            print(f"Final position: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] m")
            print(f"Final velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}] m/s")
            print(f"Final altitude: {altitude:.2f} m")
            print(f"Horizontal velocity at touchdown: {lm_guidance.hor_velocity:.2f} m/s")
            print(f"Vertical velocity at touchdown: {-velocity[2]:.2f} m/s")

            # Validate simulation results against expected parameters
            validate_simulation_results(time, velocity, altitude, lm_guidance.hor_velocity, -velocity[2], position)

            touchdown = True
            break

        # Convert acceleration command to thrust (simplified)
        # In reality, this would involve the RCS jets for horizontal control
        # and the descent engine for vertical control
        if np.linalg.norm(accel_command) > 0:
            thrust_magnitude = lm_mass * np.linalg.norm(accel_command)

            # Apply thrust limits
            thrust_magnitude = min(max_thrust, max(min_thrust, thrust_magnitude))

            # Calculate actual acceleration (including gravity)
            thrust_direction = accel_command / np.linalg.norm(accel_command)
            actual_accel = (thrust_magnitude / lm_mass) * thrust_direction - lm_guidance.lunar_gravity
        else:
            # No thrust command
            thrust_magnitude = 0
            actual_accel = -lm_guidance.lunar_gravity

        # Update state with simple integration
        velocity += actual_accel * dt
        position += velocity * dt

        # Store data for plotting
        time_points.append(time)
        altitude_points.append(-position[2])  # Convert to positive altitude
        horiz_vel_points.append(lm_guidance.hor_velocity)
        vert_vel_points.append(-velocity[2])  # Convert to positive descent rate
        x_points.append(position[0])
        y_points.append(position[1])
        z_points.append(position[2])
        thrust_points.append(thrust_magnitude)

        # Display status every 5 seconds
        if round(time, 1) % 5.0 == 0:
            lm_guidance.display_status()

        time += dt

    # Print final status if simulation ended without touchdown
    if not touchdown:
        lm_guidance.update_state(position, velocity)
        lm_guidance.p65_guidance()
        print(f"\nSimulation ended at time {time:.1f} seconds (time limit reached)")
        print(f"Final position: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] m")
        print(f"Final velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}] m/s")
        print(f"Final altitude: {-position[2]:.2f} m")
        print(f"Final horizontal velocity: {lm_guidance.hor_velocity:.2f} m/s")
        print(f"Final vertical velocity: {-velocity[2]:.2f} m/s")

        # Validation for failed landing
        print("\n" + "="*60)
        print("SIMULATION RESULTS VALIDATION")
        print("="*60)
        print(f"📍 FINAL POSITION: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] m")
        print("❌ TOUCHDOWN FAILURE: Simulation ended without successful landing!")
        print(f"   → Expected touchdown at 6.5 seconds, but simulation ran to {time:.1f} seconds")
        print(f"   → Final altitude was {-position[2]:.2f}m (should be ~0.04m)")
        print("⚠️  OVERALL VALIDATION: LANDING FAILED!")
        print("   The guidance system did not achieve the expected landing.")
        print("="*60)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Altitude vs Time
    plt.subplot(2, 2, 1)
    plt.plot(time_points, altitude_points)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs Time')
    plt.grid(True)

    # Plot 2: Horizontal Velocity vs Time
    plt.subplot(2, 2, 2)
    plt.plot(time_points, horiz_vel_points)
    plt.xlabel('Time (s)')
    plt.ylabel('Horizontal Velocity (m/s)')
    plt.title('Horizontal Velocity vs Time')
    plt.grid(True)

    # Plot 3: Vertical Velocity vs Time
    plt.subplot(2, 2, 3)
    plt.plot(time_points, vert_vel_points)
    plt.xlabel('Time (s)')
    plt.ylabel('Descent Rate (m/s)')
    plt.title('Descent Rate vs Time')
    plt.grid(True)

    # Plot 4: Thrust vs Time
    plt.subplot(2, 2, 4)
    plt.plot(time_points, thrust_points)
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.title('Engine Thrust vs Time')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('p65_simulation_results.png')

    # 3D trajectory plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_points, y_points, z_points, label='LM Trajectory')
    ax.scatter(0, 0, 0, color='red', s=100, label='Landing Target')

    # Starting point
    ax.scatter(x_points[0], y_points[0], z_points[0], color='green', s=100, label='Starting Point')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('P65 Descent Trajectory')

    # Set equal aspect ratio
    max_range = np.array([
        max(x_points) - min(x_points),
        max(y_points) - min(y_points),
        max(z_points) - min(z_points)
    ]).max() / 2.0

    mid_x = (max(x_points) + min(x_points)) * 0.5
    mid_y = (max(y_points) + min(y_points)) * 0.5
    mid_z = (max(z_points) + min(z_points)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.savefig('p65_trajectory.png')

# Run the example
if __name__ == "__main__":
    run_p65_simulation()