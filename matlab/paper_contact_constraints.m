function [margin, reaction] = paper_contact_constraints(x, pendulum, exo_torque_nm)
%PAPER_CONTACT_CONSTRAINTS C1/C3-C5 for fixed-foot human/exoskeleton dynamics.
% x contains absolute theta, speed and HUMAN torque in the pendulum's scales.
% exo_torque_nm is SI bilateral assistance. Positive margin means admissible.
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
total_torque = torque + exo_torque_nm;
acceleration = (m*g*l*sin(q) - pendulum.bFric*velocity + total_torque) / (m*l*l);
reaction.friction_n = m*l*(cos(q).*acceleration - sin(q).*velocity.^2);
reaction.normal_n = (m+pendulum.mf)*g - m*l*(sin(q).*acceleration + cos(q).*velocity.^2);
reaction.moment_nm = -total_torque + pendulum.bFric*velocity ...
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
