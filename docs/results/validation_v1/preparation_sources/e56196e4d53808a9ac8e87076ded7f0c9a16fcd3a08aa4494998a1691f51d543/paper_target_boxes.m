function [target, diagnostics] = paper_target_boxes(pendulum)
%PAPER_TARGET_BOXES Appendix II SDP/box construction for the reconciled plant.
% Centers and widths use SI units. Boxes satisfy linear and corner-contact
% constraints; enclosing a CLF ellipsoid does not prove box invariance.
p = pendulum;
stiffness = p.m*p.g*p.l;
inertia = p.m*p.l^2;
heel = pi/2-asin(p.ank/p.l);
toe = pi/2+asin((p.lf-p.ank)/p.l);
required = @(theta) stiffness*cos(theta)-p.exoTorqueSI(theta,0);
if required(heel) > p.torqueMax
    heel = fzero(@(theta) required(theta)-p.torqueMax,[heel,pi/2]);
end
if required(toe) < p.torqueMin
    toe = fzero(@(theta) required(theta)-p.torqueMin,[pi/2,toe]);
end
centers = [pi/2,toe-0.03;0,0;0,required(toe-0.03)];
B = [0;0;1];
labels = ["Ank","Toe"];
target.ankleEq = centers(:,1);
target.toeEq = centers(:,2);
for index = 1:2
    center = centers(:,index);
    % Linearize the same saturated SI assistance used by the HJB plant.
    raw = p.muGrav*stiffness*cos(center(1))+p.Kp*(pi/2-center(1));
    torque_theta = 0;
    torque_velocity = 0;
    if abs(raw) < p.motorLim
        torque_theta = -p.muGrav*stiffness*sin(center(1))-p.Kp;
        torque_velocity = -p.Kd;
    end
    A = [0,1,0; (stiffness*sin(center(1))+torque_theta)/inertia, ...
        (torque_velocity-p.bFric)/inertia,1/inertia; 0,0,0];
    Q = sdpvar(3,3,'symmetric');
    Y = sdpvar(1,3);
    L = A*Q+Q*A'+B*Y+Y'*B';
    result = optimize([Q>=eye(3),L<=0],trace(Q),sdpsettings('solver','sedumi','verbose',0));
    assert(result.problem==0,'Target SDP failed: %s',result.info);
    shape = value(Q);
    shape = (shape+shape')/2;
    gain = value(Y)/shape;
    assert(min(eig(shape))>0 && max(real(eig(A+B*gain)))<0, ...
        'Target SDP did not produce a positive-definite, stabilizing solution.');
    normals = [gain;-gain;eye(3);-eye(3)];
    slack = [p.uMax;-p.uMin;toe-center(1);0.01;p.torqueMax-center(3); ...
        center(1)-heel;0.01;center(3)-p.torqueMin];
    assert(all(slack>0),'Target center is not strictly inside the linear constraints.');
    % Eq. (48) uses distances from the selected equilibrium, not from zero.
    level = min(slack.^2./sum((normals*shape).*normals,2));
    widths = sqrt(level*diag(shape));
    % Exact linear scaling enforces every box vertex without vertex-order assumptions.
    factor = min([1;slack./(abs(normals)*widths)]);
    level = level*factor^2;
    widths = widths*factor;
    signs = 2*(dec2bin(0:7,3)-'0')-1;
    for attempt = 1:100
        vertices = center'+signs.*widths';
        states = {vertices(:,1),vertices(:,2)*p.vscale,vertices(:,3)*p.tauscale};
        if ~p.scaling
            states = num2cell(vertices,1);
        end
        margin = paper_contact_constraints(states,p,p.exoTorqueSI(vertices(:,1),vertices(:,2)));
        if min(margin)>=0
            break
        end
        widths = 0.95*widths;
        level = 0.95^2*level;
    end
    assert(min(margin)>=0,'No contact-admissible target box found.');
    label = labels(index);
    target.("theta"+label+"Range") = widths(1);
    target.("thetaDot"+label+"Range") = widths(2);
    target.("torque"+label+"Range") = widths(3);
    diagnostics(index) = struct('center_si',center,'half_widths_si',widths, ...
        'Q',shape,'K',gain,'ellipsoid_level',level, ...
        'linear_closed_loop_eigenvalues',eig(A+B*gain), ...
        'linear_constraint_max_residual',max(abs(normals)*widths-slack), ...
        'minimum_corner_contact_margin',min(margin), ...
        'nonlinear_shrink_iterations',attempt-1);
end
end
