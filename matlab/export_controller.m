function export_controller(path, time_s, torque_nm)
%EXPORT_CONTROLLER Export N-by-C HUMAN torque samples in SI units for MuJoCo.
% Columns: time_s followed by actuator-ordered torque_nm. No time offset.
% Use .mat for binary export or .csv for portable/legacy numeric arrays.
arguments
    path (1,1) string
    time_s (:,1) double {mustBeFinite, mustBeNonnegative}
    torque_nm (:,:) double {mustBeFinite}
end
assert(numel(time_s) >= 2 && all(diff(time_s) > 0), ...
    'Times must be strictly increasing with at least two samples.');
assert(size(torque_nm,1) == numel(time_s) && size(torque_nm,2) >= 1, ...
    'Torque must have one row per sample.');
assert(~isfile(path), 'Refusing to overwrite an existing controller artifact.');
[~,~,ext] = fileparts(path);
switch lower(ext)
    case '.mat'
        save(path, 'time_s', 'torque_nm', '-v7');
    case '.csv'
        writematrix([time_s, torque_nm], path);
    otherwise
        error('Use a .csv or .mat file extension.');
end
end
