function [margin, reaction] = nominal_contact_constraints(x, pendulum)
%NOMINAL_CONTACT_CONSTRAINTS Nominal source-model C1/C3-C5 in SI or scaled states.
assert(pendulum.Kp == 0 && pendulum.Kd == 0 && pendulum.muGrav == 0, ...
    'This constraint function supports the nominal no-exoskeleton model only.');
[margin,reaction] = paper_contact_constraints(x,pendulum,0);
end
