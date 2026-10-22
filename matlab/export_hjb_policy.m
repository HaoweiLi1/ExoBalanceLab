function export_hjb_policy(path, grid, value, tau, dPend)
%EXPORT_HJB_POLICY Export helperOC SI axes and a complete value time history.
% Call after HJIPDE_solve using its grid, data, tau and LiftedInvPendExo instance.
% dPend.scaling determines conversion from the authors' dimensionless system.
% value dimensions: theta x velocity x HUMAN torque x remaining-time.
arguments
    path (1,1) string
    grid struct
    value double {mustBeFinite}
    tau double {mustBeFinite}
    dPend
end
assert(~isfile(path), 'Refusing to overwrite existing policy.');
theta_rad = grid.vs{1}(:);
velocity_rad_s = grid.vs{2}(:);
human_torque_nm = grid.vs{3}(:);
time_to_go_s = tau(:);
if dPend.scaling
    velocity_rad_s = velocity_rad_s / dPend.vscale;
    human_torque_nm = human_torque_nm / dPend.tauscale;
    time_to_go_s = time_to_go_s * dPend.omega;
end
assert(size(value,1) == numel(theta_rad) && size(value,2) == numel(velocity_rad_s) ...
    && size(value,3) == numel(human_torque_nm) && size(value,4) == numel(time_to_go_s), ...
    'Export the complete HJIPDE time history; keepLast must be false.');
assert(all(diff(time_to_go_s) > 0), 'Remaining-time grid must increase.');
save(path, 'theta_rad', 'velocity_rad_s', 'human_torque_nm', 'time_to_go_s', 'value', '-v7.3');
end
