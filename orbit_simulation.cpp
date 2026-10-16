#include <iostream>
#include <cmath>   // Library for mathematical functions like sqrt() and pow()
#include <fstream> // Library for saving data to a file (for Blender)

using namespace std;

int main() {
    // Step 1: Define physical constants and initial conditions
    double G = 4.3e-3;       // Gravitational constant in pc, solar masses, and km/s units
    double mass_bh = 4.0e6;  // Mass of Sagittarius A* in solar masses (4 million Suns)
    
    // Initial position of the orbiting star (in parsecs, pc)
    double x = 0.1;          // X coordinate
    double y = 0.0;          // Y coordinate
    
    // Initial velocity of the star (in km/s converted for our scale)
    double vx = 0.0;         // Velocity in X direction
    double vy = 1000.0;      // Velocity in Y direction
    
    double dt = 0.001;       // Time step for our simulation loop
    int steps = 1000;        // Total number of simulation frames/steps

    // Step 2: Open an output file to save coordinates for Blender
    ofstream outfile("orbit_data.csv");
    outfile << "Step,X,Y\n"; // CSV Header

    // Step 3: Simulation Loop (The core engine running over time)
    for (int step = 0; step < steps; step++) {
        
        // Calculate distance (r) from the black hole at (0,0) using Pythagorean theorem
        double r = sqrt(x * x + y * y);
        
        // Prevent division by zero if the star gets too close
        if (r < 0.001) break; 

        // Calculate gravitational acceleration components using Newton's law of universal gravitation
        // a = (G * M) / r^2, broken into X and Y directions
        double accel_factor = (G * mass_bh) / (r * r * r);
        double ax = -accel_factor * x;
        double ay = -accel_factor * y;

        // Update velocity (Euler Integration method: Velocity = Acceleration * time)
        vx += ax * dt;
        vy += ay * dt;

        // Update position (Position = Velocity * time)
        x += vx * dt;
        y += vy * dt;

        // Save current coordinates to file so Blender can read them later
        outfile << step << "," << x << "," << y << "\n";
    }

    // Close the file stream
    outfile.close();
    cout << "Simulation complete! Data saved to orbit_data.csv" << endl;

    return 0;
}