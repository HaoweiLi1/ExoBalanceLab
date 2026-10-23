function export_hjb_policy(path, grid, value, tau, dPend, contract)
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
    contract struct = struct()
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
if isempty(fieldnames(contract)) && isa(dPend,'LiftedInvPendExo') ...
        && dPend.Kp == 0 && dPend.Kd == 0 && dPend.muGrav == 0
    % The nominal source law matches the paper plant exactly after SI conversion.
    contract.model = 'paper';
    contract.exo = 'none';
    contract.profile = struct('mass_kg',dPend.m+dPend.mf,'height_m',dPend.l/0.575, ...
        'torque_min_nm',dPend.torqueMin,'torque_max_nm',dPend.torqueMax, ...
        'rate_min_nm_s',dPend.uMin,'rate_max_nm_s',dPend.uMax, ...
        'damping_nm_s',dPend.bFric,'friction',1,'gravity_m_s2',dPend.g);
end
if isempty(fieldnames(contract))
    save(path,'theta_rad','velocity_rad_s','human_torque_nm','time_to_go_s','value','-v7.3');
else
    contract_json = jsonencode(contract);
    save(path,'theta_rad','velocity_rad_s','human_torque_nm','time_to_go_s','value', ...
        'contract_json','-v7.3');
end
end
