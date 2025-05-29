import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from descent import LunarDescentGuidance

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