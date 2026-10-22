function summary = prepare_source_target_override(reference_dir, output_dir)
%PREPARE_SOURCE_TARGET_OVERRIDE Isolate the released entry's ankle-box bounds.
% Derive one OM/GC/forward 225-grid input from an existing prepared experiment.
% Angles, velocities and human torques use rad, rad/s and Nm in the metadata.
% The stored nondimensional fields retain their original scales and obstacles.
arguments
    reference_dir (1,1) string
    output_dir (1,1) string
end
assert(~isfolder(output_dir) && ~isfile(output_dir), 'Refusing to overwrite an experiment.');
problem = load(fullfile(reference_dir, 'problem.mat'));
metadata = jsondecode(problem.problem_json);
summary = metadata.summary;
assert(strcmp(summary.profile, 'OM') && strcmp(summary.exo, 'gc') && ...
    strcmp(summary.direction, 'forward') && all(summary.requested_resolution == 225), ...
    'This recorded diagnostic requires the OM/GC/forward 225-grid input.');
target = summary.target;
previous_velocity = target.thetaDotAnkRange;
previous_torque = target.torqueAnkRange;
assert(previous_velocity <= 0.01 && previous_torque <= 1, ...
    'The recorded source override must contain the original ankle box.');
target.thetaDotAnkRange = 0.01;
target.torqueAnkRange = 1;
scale = problem.state_scale(:);
center = target.ankleEq(:).*scale;
width = [target.thetaAnkRange; target.thetaDotAnkRange; target.torqueAnkRange].*scale;
state_axes = {reshape(problem.theta_rad*scale(1), [], 1, 1), ...
    reshape(problem.velocity_rad_s*scale(2), 1, [], 1), ...
    reshape(problem.human_torque_nm*scale(3), 1, 1, [])};
ankle_value = -inf;
for dimension = 1:3
    ankle_value = max(ankle_value, center(dimension)-width(dimension)-state_axes{dimension});
    ankle_value = max(ankle_value, state_axes{dimension}-(center(dimension)+width(dimension)));
end
% Union with the containing ankle box preserves the original toe field exactly.
previous_initial = problem.initial_value;
problem.initial_value = max(min(previous_initial, ankle_value), problem.obstacle_value);
assert(all(isfinite(problem.initial_value), 'all'));
assert(all(problem.initial_value <= previous_initial, 'all'));
assert(all(problem.initial_value >= problem.obstacle_value, 'all'));
summary.target = target;
summary.ankle_target_nodes = nnz(max(ankle_value, problem.obstacle_value) < 0);
summary.initial_target_nodes = nnz(problem.initial_value < 0);
summary.scope = 'Diagnostic source target override on the reconciled GC plant; not an unchanged author run.';
source_url = "https://github.com/Raquelometer/AnkleExoBRS/blob/" + ...
    "517ab41b7767c5ab0e856a8885b118bd739a5011/computeAnkleExoBRS.m#L45";
summary.target_override = struct( ...
    'reference_prepared', reference_dir, ...
    'source_url', source_url, ...
    'previous_velocity_half_width_rad_s', previous_velocity, ...
    'previous_human_torque_half_width_nm', previous_torque, ...
    'new_velocity_half_width_rad_s', target.thetaDotAnkRange, ...
    'new_human_torque_half_width_nm', target.torqueAnkRange, ...
    'initial_field_changed_nodes', nnz(problem.initial_value ~= previous_initial), ...
    'ankle_nodes_before_obstacle_clipping', nnz(ankle_value < 0));
metadata.summary = summary;
problem.problem_json = jsonencode(metadata);
mkdir(output_dir);
save(fullfile(output_dir, 'problem.mat'), '-struct', 'problem', '-v7.3');
copyfile(mfilename('fullpath') + ".m", fullfile(output_dir, 'prepare_source_target_override.m'));
file = fopen(fullfile(output_dir, 'summary.json'), 'w');
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(summary, PrettyPrint=true));
end
