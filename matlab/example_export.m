% Export an ordinary controller trajectory for offline MuJoCo replay.
% Run from the repository root: addpath('matlab'); run('matlab/example_export.m')
time_s = (0:0.001:6)';
torque_nm = 8 * sin(2*pi*0.5*time_s) .* exp(-time_s);
if ~isfolder('logs'), mkdir('logs'); end
export_controller('logs/matlab_example.mat', time_s, torque_nm);
% Python: exobalance run --trajectory logs/matlab_example.mat --lean 0 --velocity 0
