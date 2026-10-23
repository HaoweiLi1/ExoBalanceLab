%% System Parameters
% Anthropometric parameters (from the paper and ankle_lqr_simulation.m)
M_total = 60;                   % Total body mass (kg)
H_total = 1.59;                 % Total height (m)
m_feet = 2 * 0.0145 * M_total;  % Mass of feet (kg)
m_body = M_total - m_feet;      % Mass of body (kg)
l_COM = 0.575 * H_total;        % Distance to COM from ankle (m)
I = m_body * l_COM^2;           % Moment of inertia (kg·m²)
g = 9.81;                       % Gravity (m/s²)
b = 2.5;                        % Damping coefficient (Nm·s/rad)

% Control constraints
MT_df = 22;                     % Maximum dorsiflexion torque (Nm)
MT_pf = -88;                    % Maximum plantarflexion torque (Nm)
MRTD_df = 148;                  % Maximum rate of torque development - dorsiflexion (Nm/s)
MRTD_pf = 389;                  % Maximum rate of torque development - plantarflexion (Nm/s)

% Initial conditions
theta_0 = -0.03;                % Initial ankle angle (rad)
theta_dot_0 = 0.3;              % Initial ankle angular velocity (rad/s)

% Target position
theta_target = 0.0;             % Target ankle angle (rad)

% Simulation parameters
% FIXED: Reduced time resolution to make the problem more tractable
dt = 0.0025;                      % Time step (s) - increased from 0.0005
T_sim = 6;                      % Simulation duration (s)
N = round(T_sim/dt) + 1;        % Number of time steps

% Foot geometry and friction parameters
l_heel = 0.05 * H_total;        % Distance from ankle to heel (m)
l_toe = 0.15 * H_total;         % Distance from ankle to toe (m)
mu = 0.99;                       % Friction coefficient

%% Setup for Direct Collocation Optimal Control
% Based on the "Discretized Optimal Trajectory" tutorial

% MODIFICATION 2: Update optimization variables
% Create optimization variables - 3D lifted system approach
theta = optimvar('theta', N, 1);
theta_dot = optimvar('theta_dot', N, 1);
tau = optimvar('tau', N, 1, 'LowerBound', MT_pf, 'UpperBound', MT_df);
u = optimvar('u', N-1, 1, 'LowerBound', -MRTD_pf, 'UpperBound', MRTD_df);

% Create optimization problem
ankleProb = optimproblem;

% MODIFICATION 2: Update objective function to minimize squared torque rate
ankleProb.Objective = sum(u.^2) * dt;

% MODIFICATION 2: Define equations of motion constraints for 3D system
% Position dynamics (trapezoidal rule)
p_dynamics = optimconstr(N-1);
for i = 2:N
    p_dynamics(i-1) = theta(i) == theta(i-1) + dt * (theta_dot(i-1) + theta_dot(i)) / 2;
end

% Velocity dynamics
v_dynamics = optimconstr(N-1);
for i = 2:N
    v_dynamics(i-1) = theta_dot(i) == theta_dot(i-1) + dt * ((tau(i-1)/I) - (b/I)*theta_dot(i-1) + (m_body*g*l_COM/I)*sin(theta(i-1)));
end

% MODIFICATION 2: Add torque dynamics (torque is now a state)
tau_dynamics = optimconstr(N-1);
for i = 2:N
    tau_dynamics(i-1) = tau(i) == tau(i-1) + dt * u(i-1);
end

% MODIFICATION 2: Update initial conditions to include torque
initialcons = [
    theta(1) == theta_0; 
    theta_dot(1) == theta_dot_0;
    tau(1) == 0  % Initial torque is zero
];

% Final conditions (want to reach target and stop)
finalcons = [
    theta(N) == theta_target; 
    theta_dot(N) == 0
    % tau(N) is left free
];

% Add angle range constraint to prevent extreme angles
max_angle = 30 * pi/180;  % 30 degrees in radians
angle_range_upper = optimconstr(N);
angle_range_lower = optimconstr(N);
for i = 1:N
    angle_range_upper(i) = theta(i) <= max_angle;
    angle_range_lower(i) = theta(i) >= -max_angle;
end

% Add all constraints to the problem
ankleProb.Constraints.v_dynamics = v_dynamics;
ankleProb.Constraints.p_dynamics = p_dynamics;
ankleProb.Constraints.tau_dynamics = tau_dynamics;  % MODIFICATION 2: Add torque dynamics
ankleProb.Constraints.initialcons = initialcons;
ankleProb.Constraints.finalcons = finalcons;
ankleProb.Constraints.angle_range_upper = angle_range_upper;
ankleProb.Constraints.angle_range_lower = angle_range_lower;

% MODIFICATION 2: Create an initial guess for the variables
x0.theta = linspace(theta_0, theta_target, N)';
x0.theta_dot = linspace(theta_dot_0, 0, N)';
x0.tau = zeros(N, 1);
x0.u = zeros(N-1, 1);

% FIXED: Create solver options and specify solver explicitly
options = optimoptions('fmincon', 'Display', 'iter', ...
                      'Algorithm', 'interior-point', ...
                      'MaxFunctionEvaluations', 20000, ... % Increased from 10000
                      'MaxIterations', 3000, ...           % Increased from 1000
                      'OptimalityTolerance', 1e-6);

% FIXED: Explicitly specify fmincon as the solver
[sol, fval, exitflag, output] = solve(ankleProb, x0, 'Solver', 'fmincon', 'Options', options);

% Check if a solution was found
if exitflag > 0
    disp('Optimal solution found!');
else
    warning('Problem not solved successfully. Exitflag: %d', exitflag);
    disp(output.message);
end

%% Extract solution and propagate dynamics if necessary
if exitflag > 0
    % MODIFICATION 2: Extract the solution including torque and torque rate
    theta_sol = sol.theta;
    theta_dot_sol = sol.theta_dot;
    tau_sol = sol.tau;
    u_sol = sol.u;
    
    % Create time vector
    t = (0:N-1) * dt;
else
    % If optimization fails, fallback to a simple control strategy
    disp('Using fallback control strategy...');
    
    % MODIFICATION 2: Update fallback strategy to use 3D system
    % Allocate arrays
    theta_sol = zeros(N, 1);
    theta_dot_sol = zeros(N, 1);
    tau_sol = zeros(N, 1);
    u_sol = zeros(N-1, 1);
    
    % Set initial conditions
    theta_sol(1) = theta_0;
    theta_dot_sol(1) = theta_dot_0;
    tau_sol(1) = 0;  % Initial torque is zero
    
    % Simple PD control loop
    Kp = 200;
    Kd = 50;
    
    for i = 1:N-1
        % Calculate desired torque with PD feedback
        error = theta_target - theta_sol(i);
        error_dot = -theta_dot_sol(i);
        
        desired_torque = Kp * error + Kd * error_dot;
        
        % Calculate torque rate (RTD) to reach desired torque
        u_sol(i) = (desired_torque - tau_sol(i)) / dt;
        
        % Apply rate limits
        u_sol(i) = min(max(u_sol(i), -MRTD_pf), MRTD_df);
        
        % Update torque
        tau_sol(i+1) = tau_sol(i) + dt * u_sol(i);
        
        % Apply torque limits
        tau_sol(i+1) = min(max(tau_sol(i+1), MT_pf), MT_df);
        
        % Simple Euler integration
        theta_dot_sol(i+1) = theta_dot_sol(i) + dt * ((tau_sol(i)/I) - (b/I)*theta_dot_sol(i) + (m_body*g*l_COM/I)*sin(theta_sol(i)));
        theta_sol(i+1) = theta_sol(i) + dt * theta_dot_sol(i);
    end
    
    % Create time vector
    t = (0:N-1) * dt;
end

%% Compute Derived Quantities
% Calculate the gravity torque
gravity_torque = m_body * g * l_COM * sin(theta_sol);
% FIXED: Ensure arrays are properly aligned for export
fprintf('Checking dimensions before export:\n');
fprintf('Size of t: %d x %d\n', size(t, 1), size(t, 2));
fprintf('Size of tau_sol: %d x %d\n', size(tau_sol, 1), size(tau_sol, 2));

% Make sure both vectors are row vectors for proper concatenation
t_row = reshape(t, 1, []);          % Ensure t is a row vector
tau_row = reshape(tau_sol, 1, []);  % Ensure tau_sol is a row vector

% Ensure they have the same length
min_length = min(length(t_row), length(tau_row));
t_row = t_row(1:min_length);
tau_row = tau_row(1:min_length);

% Now create the data for export - stacking rows vertically
torque_data = [t_row; tau_row];

% Check final dimensions
fprintf('Size of torque_data: %d x %d\n', size(torque_data, 1), size(torque_data, 2));

% Save to CSV file
csv_filename = 'optimal_ankle_torque_trajectory.csv';
writematrix(torque_data, csv_filename);
fprintf('Torque trajectory saved to: %s\n', csv_filename);

%% Visualization
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Ankle Position
subplot(2, 2, 1);
plot(t, theta_sol*180/pi, 'LineWidth', 2);
hold on;
plot(t, ones(size(t))*theta_target*180/pi, 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Ankle Angle (deg)');
title('Ankle Joint Position');
legend('Actual Position', 'Target Position');

% Plot 2: Ankle Velocity
subplot(2, 2, 2);
plot(t, theta_dot_sol*180/pi, 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (deg/s)');
title('Ankle Joint Velocity');

% MODIFICATION 2: Update torque plot to use tau_sol
subplot(2, 2, 3);
plot(t, tau_sol, 'LineWidth', 2);
hold on;
plot(t, gravity_torque, 'g--', 'LineWidth', 1.5);
plot(t, ones(size(t))*MT_df, 'k--');
plot(t, ones(size(t))*MT_pf, 'k--');
grid on;
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Ankle Joint Torque');
legend('Applied Torque', 'Gravity Torque', 'Physiological Limits');

% MODIFICATION 2: Update RTD plot to use u_sol directly
subplot(2, 2, 4);
plot(t(1:end-1), u_sol, 'LineWidth', 2);
hold on;
plot(t, ones(size(t))*MRTD_df, 'r--', 'LineWidth', 1);
plot(t, -ones(size(t))*MRTD_pf, 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('RTD (Nm/s)');
title('Rate of Torque Development');
legend('Applied RTD', 'RTD Limits');

% Adjust layout
sgtitle('Optimal Ankle Balance Control with 3D Lifted System (No Foot-Ground Constraints)', 'FontSize', 16);
set(gcf, 'Color', 'w');

%% Additional Analysis: Phase Plane Trajectory
figure('Position', [100, 550, 600, 500]);
plot(theta_sol*180/pi, theta_dot_sol*180/pi, 'b-', 'LineWidth', 2);
hold on;
plot(theta_sol(1)*180/pi, theta_dot_sol(1)*180/pi, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
plot(theta_target*180/pi, 0, 'gx', 'MarkerSize', 8, 'LineWidth', 2);
grid on;
xlabel('Ankle Angle (deg)');
ylabel('Angular Velocity (deg/s)');
title('Phase Plane Trajectory');
legend('Trajectory', 'Initial State', 'Target State');

%% MODIFICATION 2: Add a new plot for Torque vs Time and Torque Rate vs Time
figure('Position', [700, 550, 800, 400]);

subplot(1, 2, 1);
plot(t, tau_sol, 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Ankle Torque');

subplot(1, 2, 2);
% FIXED: Ensure we only plot u_sol with compatible t values
plot(t(1:length(u_sol)), u_sol, 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Torque Rate (Nm/s)');
title('Rate of Torque Development');

%% Display Summary Statistics
fprintf('\n---- Simulation Results ----\n');
fprintf('Final ankle position: %.4f deg\n', theta_sol(end)*180/pi);
fprintf('Position error: %.4f deg\n', (theta_sol(end) - theta_target)*180/pi);
fprintf('Maximum ankle angle: %.4f deg\n', max(abs(theta_sol))*180/pi);
fprintf('Maximum angular velocity: %.4f deg/s\n', max(abs(theta_dot_sol))*180/pi);
fprintf('Maximum torque: %.4f Nm\n', max(abs(tau_sol)));
fprintf('Maximum RTD: %.4f Nm/s\n', max(abs(u_sol)));
fprintf('--------------------------\n');