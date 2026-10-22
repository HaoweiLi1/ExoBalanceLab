function torque_nm = standing_pd(time_s, state)
%STANDING_PD Example OF human controller for the persistent Engine adapter.
% state = [q; qdot; applied human torque]; q=theta-pi/2, units rad/rad/s/Nm.
% This example assumes no exoskeleton. MuJoCo enforces human MT/MRTD limits.
arguments
    time_s (1,1) double {mustBeFinite, mustBeNonnegative}
    state (3,1) double {mustBeFinite}
end
m = 60 * (1 - 2*0.0145);
l = 1.59 * 0.575;
kp = m * 9.81 * l;
inertia = m * l^2;
torque_nm = -1.8 * kp * state(1) - 0.8 * sqrt(inertia * kp) * state(2);
end
