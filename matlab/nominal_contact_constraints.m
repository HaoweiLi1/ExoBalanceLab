function [margin, reaction] = nominal_contact_constraints(x, pendulum)
%NOMINAL_CONTACT_CONSTRAINTS C1/C3-C5 for the authors' no-exoskeleton plant.
% x is a cell array {absolute theta, scaled speed, scaled human torque}.
% Positive margin means admissible. Reaction fields use SI units. Moment is
% the ground moment about the ankle's ground projection, so CoP=moment/Fz.
assert(pendulum.Kp == 0 && pendulum.Kd == 0 && pendulum.muGrav == 0, ...
    'This constraint function supports the nominal no-exoskeleton model only.');
q = x{1} - pi/2;
velocity = x{2};
torque = x{3};
if pendulum.scaling
    velocity = velocity / pendulum.vscale;
    torque = torque / pendulum.tauscale;
end
m = pendulum.m;
l = pendulum.l;
g = pendulum.g;
acceleration = (m*g*l*sin(q) - pendulum.bFric*velocity + torque) / (m*l*l);
reaction.friction_n = m*l*(cos(q).*acceleration - sin(q).*velocity.^2);
reaction.normal_n = (m+pendulum.mf)*g - m*l*(sin(q).*acceleration + cos(q).*velocity.^2);
reaction.moment_nm = -torque + pendulum.bFric*velocity ...
    + pendulum.mf*g*pendulum.c - pendulum.b*reaction.friction_n;
weight = (m+pendulum.mf)*g;
heel = -pendulum.ank;
toe = pendulum.lf-pendulum.ank;
% Moment inequalities avoid division by vanishing normal force at contact loss.
margin = min(reaction.normal_n/weight, ...
    (reaction.moment_nm-heel*reaction.normal_n)/(weight*pendulum.lf));
margin = min(margin, (toe*reaction.normal_n-reaction.moment_nm)/(weight*pendulum.lf));
margin = min(margin, (reaction.normal_n-abs(reaction.friction_n))/weight); % paper mu=1
margin = min(margin, (torque-pendulum.torqueMin)/(m*g*l));
margin = min(margin, (pendulum.torqueMax-torque)/(m*g*l));
end
