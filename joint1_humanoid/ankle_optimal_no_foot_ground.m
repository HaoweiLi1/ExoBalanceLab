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
theta_0 = -0.02;                % Initial ankle angle (rad)
theta_dot_0 = 0.2;              % Initial ankle angular velocity (rad/s)

% Target position
theta_target = 0.0;             % Target ankle angle (rad)

% Simulation parameters
dt = 0.0005;                    % Time step (s)
T_sim = 0.5;                      % Simulation duration (s)
N = round(T_sim/dt) + 1;        % Number of time steps

% Foot geometry and friction parameters
l_heel = 0.05 * H_total;        % Distance from ankle to heel (m)
l_toe = 0.15 * H_total;         % Distance from ankle to toe (m)
mu = 0.8;                       % Friction coefficient

%% Setup for Direct Collocation Optimal Control
% Based on the "Discretized Optimal Trajectory" tutorial

% Create optimization variables
theta = optimvar('theta', N, 1);
theta_dot = optimvar('theta_dot', N, 1);
a = optimvar('a', N-1, 1, 'LowerBound', MT_pf, 'UpperBound', MT_df);
s = optimvar('s', N-1, 1, 'LowerBound', 0);

% Create optimization problem
ankleProb = optimproblem;

% Define objective function: minimize control effort
ankleProb.Objective = sum(s) * dt;

% Add constraint for the norm of acceleration (using s as helper variable)
% We can't use abs() directly with optimization variables
% Instead, we create two inequality constraints equivalent to |a(i)| <= s(i)
acons_pos = optimconstr(N-1);
acons_neg = optimconstr(N-1);
for i = 1:N-1
    acons_pos(i) = a(i) <= s(i);     % a(i) ≤ s(i)
    acons_neg(i) = -a(i) <= s(i);    % -a(i) ≤ s(i)
end

% Define equations of motion constraints
% θ̇(i) = θ̇(i-1) + dt*(a(i-1) + g)
v_dynamics = optimconstr(N-1);
for i = 2:N
    v_dynamics(i-1) = theta_dot(i) == theta_dot(i-1) + dt * ((a(i-1)/I) - (b/I)*theta_dot(i-1) + (m_body*g*l_COM/I)*sin(theta(i-1)));
end

% θ(i) = θ(i-1) + dt*(θ̇(i-1) + θ̇(i))/2
p_dynamics = optimconstr(N-1);
for i = 2:N
    p_dynamics(i-1) = theta(i) == theta(i-1) + dt * (theta_dot(i-1) + theta_dot(i)) / 2;
end

% Rate of torque development constraints
rtd_upper_constraints = optimconstr(N-2);
rtd_lower_constraints = optimconstr(N-2);
for i = 1:N-2
    rtd_upper_constraints(i) = (a(i+1) - a(i))/dt <= MRTD_df;  % Upper bound
    rtd_lower_constraints(i) = (a(i+1) - a(i))/dt >= -MRTD_pf; % Lower bound
end

% Initial conditions
initialcons = [theta(1) == theta_0; theta_dot(1) == theta_dot_0];

% Final conditions (want to reach target and stop)
finalcons = [theta(N) == theta_target; theta_dot(N) == 0];

% MODIFICATION 1: Comment out foot-ground contact constraints
% Define state-dependent bound functions
% For tipping constraints:
% h_upper_fun = @(theta_val) m_body * g * cos(theta_val) * l_toe;
% h_lower_fun = @(theta_val) -m_body * g * cos(theta_val) * l_heel;
% For friction constraint:
% friction_bound_fun = @(theta_val) mu * m_body * g * min(l_toe, l_heel);

% Create SEPARATE constraint arrays for upper and lower bounds
% foot_upper_constraints = optimconstr(N-1);
% foot_lower_constraints = optimconstr(N-1);

% for i = 1:N-1
%     % Net ankle torque (human only in current model)
%     ankle_torque = a(i);
%     
%     % Using fcn2optimexpr to handle nonlinear expressions
%     tau_upper_tipping = fcn2optimexpr(h_upper_fun, theta(i));
%     tau_lower_tipping = fcn2optimexpr(h_lower_fun, theta(i));
%     tau_friction_bound = fcn2optimexpr(friction_bound_fun, theta(i));
%     
%     % Apply most restrictive constraint (tipping vs. friction)
%     % For friction, we need both upper and lower bounds
%     tau_upper = fcn2optimexpr(@(u1, u2) min(u1, u2), tau_upper_tipping, tau_friction_bound);
%     tau_lower = fcn2optimexpr(@(l1, l2) max(l1, l2), tau_lower_tipping, -tau_friction_bound);
%     
%     % Add constraints to the problem as SEPARATE constraints
%     foot_upper_constraints(i) = ankle_torque <= tau_upper;
%     foot_lower_constraints(i) = ankle_torque >= tau_lower;
% end

% Add angle range constraint to prevent extreme angles
max_angle = 30 * pi/180;  % 30 degrees in radians
angle_range_upper = optimconstr(N);
angle_range_lower = optimconstr(N);
for i = 1:N
    angle_range_upper(i) = theta(i) <= max_angle;
    angle_range_lower(i) = theta(i) >= -max_angle;
end

% Add all constraints to the problem
ankleProb.Constraints.acons_pos = acons_pos;
ankleProb.Constraints.acons_neg = acons_neg;
ankleProb.Constraints.v_dynamics = v_dynamics;
ankleProb.Constraints.p_dynamics = p_dynamics;
ankleProb.Constraints.rtd_upper_constraints = rtd_upper_constraints;
ankleProb.Constraints.rtd_lower_constraints = rtd_lower_constraints;
ankleProb.Constraints.initialcons = initialcons;
ankleProb.Constraints.finalcons = finalcons;
% MODIFICATION 1: Comment out foot-ground constraints
% ankleProb.Constraints.foot_upper_constraints = foot_upper_constraints;
% ankleProb.Constraints.foot_lower_constraints = foot_lower_constraints;
ankleProb.Constraints.angle_range_upper = angle_range_upper;
ankleProb.Constraints.angle_range_lower = angle_range_lower;

% Create an initial guess for the variables
x0.theta = linspace(theta_0, theta_target, N)';
x0.theta_dot = linspace(theta_dot_0, 0, N)';
x0.a = zeros(N-1, 1);
x0.s = ones(N-1, 1);

% Set options for the solver
options = optimoptions('fmincon', 'Display', 'iter', ...
                      'Algorithm', 'interior-point', ...
                      'MaxFunctionEvaluations', 10000, ...
                      'MaxIterations', 1000, ...
                      'OptimalityTolerance', 1e-6);

% Solve the problem
[sol, fval, exitflag, output] = solve(ankleProb, x0, 'Options', options);

% Check if a solution was found
if exitflag > 0
    disp('Optimal solution found!');
else
    warning('Problem not solved successfully. Exitflag: %d', exitflag);
    disp(output.message);
end

%% Extract solution and propagate dynamics if necessary
if exitflag > 0
    % Extract the solution
    theta_sol = sol.theta;
    theta_dot_sol = sol.theta_dot;
    a_sol = sol.a;
    s_sol = sol.s;
    
    % Create time vector
    t = (0:N-1) * dt;
else
    % If optimization fails, fallback to a simple control strategy
    disp('Using fallback control strategy...');
    
    % Allocate arrays
    theta_sol = zeros(N, 1);
    theta_dot_sol = zeros(N, 1);
    a_sol = zeros(N-1, 1);
    
    % Set initial conditions
    theta_sol(1) = theta_0;
    theta_dot_sol(1) = theta_dot_0;
    
    % Simple PD control loop
    Kp = 200;
    Kd = 50;
    
    for i = 1:N-1
        % Calculate control with PD feedback
        error = theta_target - theta_sol(i);
        error_dot = -theta_dot_sol(i);
        
        a_sol(i) = Kp * error + Kd * error_dot;
        
        % Apply torque limits
        a_sol(i) = min(max(a_sol(i), MT_pf), MT_df);
        
        % MODIFICATION 1: Comment out foot-ground constraints check
        % Apply foot-ground constraints
        % Calculate bounds for this state
        % tau_upper_tipping = h_upper_fun(theta_sol(i));
        % tau_lower_tipping = h_lower_fun(theta_sol(i));
        % tau_friction_bound = friction_bound_fun(theta_sol(i));
        
        % Combined bounds
        % tau_upper = min(tau_upper_tipping, tau_friction_bound);
        % tau_lower = max(tau_lower_tipping, -tau_friction_bound);
        
        % Apply the constraints
        % a_sol(i) = min(max(a_sol(i), tau_lower), tau_upper);
        
        % Simple Euler integration
        theta_dot_sol(i+1) = theta_dot_sol(i) + dt * ((a_sol(i)/I) - (b/I)*theta_dot_sol(i) + (m_body*g*l_COM/I)*sin(theta_sol(i)));
        theta_sol(i+1) = theta_sol(i) + dt * theta_dot_sol(i);
    end
    
    % Create time vector
    t = (0:N-1) * dt;
end

%% Compute Derived Quantities
% Calculate torque rate of change for analysis
rtd_sol = zeros(N-1, 1);
for i = 1:N-2
    rtd_sol(i) = (a_sol(i+1) - a_sol(i)) / dt;
end
rtd_sol(N-1) = rtd_sol(N-2);  % Repeat the last value for plotting

% Calculate the gravity torque
gravity_torque = m_body * g * l_COM * sin(theta_sol);

% MODIFICATION 1: Comment out foot-ground constraint bounds calculation
% Calculate the foot-ground constraint bounds along the trajectory
% tipping_upper_bound = zeros(N, 1);
% tipping_lower_bound = zeros(N, 1);
% friction_bound = zeros(N, 1);
% for i = 1:N
%     tipping_upper_bound(i) = h_upper_fun(theta_sol(i));
%     tipping_lower_bound(i) = h_lower_fun(theta_sol(i));
%     friction_bound(i) = friction_bound_fun(theta_sol(i));
% end

% Calculate combined bounds
% upper_bound = min(tipping_upper_bound, friction_bound);
% lower_bound = max(tipping_lower_bound, -friction_bound);

%% Export torque trajectory to CSV
% Create a table with time and applied torque
torque_data = [t'; [a_sol; a_sol(end)]'];

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

% Plot 3: Torque (MODIFIED to remove foot-ground bounds)
subplot(2, 2, 3);
plot(t(1:N-1), a_sol, 'LineWidth', 2);
hold on;
plot(t, gravity_torque, 'g--', 'LineWidth', 1.5);
% MODIFICATION 1: Comment out foot-ground constraint bounds plotting
% plot(t, upper_bound, 'r-.', 'LineWidth', 1.5);
% plot(t, lower_bound, 'r-.', 'LineWidth', 1.5);
plot(t, ones(size(t))*MT_df, 'k--');
plot(t, ones(size(t))*MT_pf, 'k--');
grid on;
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Ankle Joint Torque');
% MODIFICATION 1: Update legend without foot-ground bounds
legend('Applied Torque', 'Gravity Torque', 'Physiological Limits');

% Plot 4: Rate of Torque Development
subplot(2, 2, 4);
plot(t(1:N-1), rtd_sol, 'LineWidth', 2);
hold on;
plot(t, ones(size(t))*MRTD_df, 'r--', 'LineWidth', 1);
plot(t, -ones(size(t))*MRTD_pf, 'r--', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('RTD (Nm/s)');
title('Rate of Torque Development');
legend('Applied RTD', 'RTD Limits');

% Adjust layout
sgtitle('Optimal Ankle Balance Control Simulation with Foot-Ground Constraints Removed', 'FontSize', 16);
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

% MODIFICATION 1: Remove foot-ground constraints figure
% %% Additional Analysis: Foot-Ground Constraints
% figure('Position', [700, 550, 700, 500]);
% plot(t, tipping_upper_bound, 'b-', 'LineWidth', 1.5);
% hold on;
% plot(t, tipping_lower_bound, 'b-', 'LineWidth', 1.5);
% plot(t, friction_bound, 'g--', 'LineWidth', 1.5);
% plot(t, -friction_bound, 'g--', 'LineWidth', 1.5);
% plot(t, upper_bound, 'r-', 'LineWidth', 2);
% plot(t, lower_bound, 'r-', 'LineWidth', 2);
% plot(t(1:N-1), a_sol, 'k-', 'LineWidth', 2);
% grid on;
% xlabel('Time (s)');
% ylabel('Torque (Nm)');
% title('Foot-Ground Constraint Analysis');
% legend('Tipping Upper Bound', 'Tipping Lower Bound', 'Friction Upper Bound', 'Friction Lower Bound', 'Combined Upper Bound', 'Combined Lower Bound', 'Applied Torque');

%% Display Summary Statistics
fprintf('\n---- Simulation Results ----\n');
fprintf('Final ankle position: %.4f deg\n', theta_sol(end)*180/pi);
fprintf('Position error: %.4f deg\n', (theta_sol(end) - theta_target)*180/pi);
fprintf('Maximum ankle angle: %.4f deg\n', max(abs(theta_sol))*180/pi);
fprintf('Maximum angular velocity: %.4f deg/s\n', max(abs(theta_dot_sol))*180/pi);
fprintf('Maximum torque: %.4f Nm\n', max(abs(a_sol)));
fprintf('Maximum RTD: %.4f Nm/s\n', max(abs(rtd_sol)));
fprintf('--------------------------\n');

% MODIFICATION 1: Remove foot-ground constraint check
% Check if solution respects foot-ground constraints
% torque_within_bounds = all(a_sol <= upper_bound(1:N-1) & a_sol >= lower_bound(1:N-1));
% fprintf('Solution respects foot-ground constraints: %s\n', mat2str(torque_within_bounds));

% Check which constraint is more restrictive (tipping or friction)
% tipping_more_restrictive_upper = sum(tipping_upper_bound < friction_bound);
% tipping_more_restrictive_lower = sum(tipping_lower_bound > -friction_bound);
% fprintf('Tipping constraint more restrictive than friction: %.1f%% (upper), %.1f%% (lower)\n', ...
%     100*tipping_more_restrictive_upper/N, 100*tipping_more_restrictive_lower/N);