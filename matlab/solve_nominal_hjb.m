function summary = solve_nominal_hjb(author_root, dependency_root, output_dir, options)
%SOLVE_NOMINAL_HJB Run the authors' no-exoskeleton HJB model with C1/C3-C5.
% Uses original model/target MAT files and unchanged LiftedInvPendExo dynamics.
% Contact constraints are independently reconstructed; this is not an exact
% reproduction of the missing upstream friction implementation or Table II.
arguments
    author_root (1,1) string
    dependency_root (1,1) string
    output_dir (1,1) string
    options.profile (1,1) string {mustBeMember(options.profile,["YF","OF","YM","OM","wOF"])} = "OF"
    options.direction (1,1) string {mustBeMember(options.direction,["forward","backward"])} = "forward"
    options.resolution (3,1) double {mustBeInteger,mustBePositive} = [51;225;51]
    options.duration_s (1,1) double {mustBePositive,mustBeFinite} = 1.4
    options.output_step_s (1,1) double {mustBePositive,mustBeFinite} = 0.1
    options.method (1,1) string {mustBeMember(options.method,["none","minVWithV0"])} = "none"
end
assert(~isfolder(output_dir) && ~isfile(output_dir), 'Refusing to overwrite an HJB run.');
assert(all(options.resolution >= 3), 'Each grid dimension needs at least three points.');
addpath(genpath(fullfile(dependency_root, 'helperOC')));
addpath(genpath(fullfile(dependency_root, 'ToolboxLS', 'ToolboxLS', 'Kernel')));
addpath(fullfile(author_root, 'Dynamics', 'DynUtils'));
addpath(fullfile(author_root, 'Models', 'ModelUtils'));
model_file = fullfile(author_root, 'Models', options.profile + "_model.mat");
target_file = fullfile(author_root, 'TargetSets', 'computedSets', ...
    "Params_" + options.profile + "_noExo_100_pctMT_100_pctRTD.mat");
model_data = load(model_file, 'model');
target_data = load(target_file, 'problemParams');
% MATLAB discovers the external dynamics/optCtrl methods only in an @ folder.
% Stage unchanged source into the run, leaving the authors' checkout untouched.
mkdir(output_dir);
class_root = fullfile(output_dir,'source');
mkdir(class_root);
copyfile(fullfile(author_root,'Dynamics','LiftedInvPendExo'), ...
    fullfile(class_root,'@LiftedInvPendExo'));
addpath(class_root);
copyfile(model_file, fullfile(output_dir,'source_model.mat'));
copyfile(target_file, fullfile(output_dir,'source_target.mat'));
pendulum = LiftedInvPendExo([pi/2;0;0], model_data.model, target_data.problemParams);
if options.direction == "forward"
    lower = [0.65;-0.02*pendulum.vscale;pendulum.torqueMin*pendulum.tauscale];
    upper = [1.8;2.85*pendulum.vscale;pendulum.torqueMax*pendulum.tauscale];
else
    lower = [1.51;-2.85*pendulum.vscale;pendulum.torqueMin*pendulum.tauscale];
    upper = [2.56;0.02*pendulum.vscale;pendulum.torqueMax*pendulum.tauscale];
end
grid = createGrid(lower, upper, options.resolution);
target = target_data.problemParams.targetSets;
ankle_width = [target.thetaAnkRange;target.thetaDotAnkRange*pendulum.vscale; ...
    target.torqueAnkRange*pendulum.tauscale];
toe_width = [target.thetaToeRange;target.thetaDotToeRange*pendulum.vscale; ...
    target.torqueToeRange*pendulum.tauscale];
scale = [1;pendulum.vscale;pendulum.tauscale];
ankle = target.ankleEq(:).*scale;
toe = target.toeEq(:).*scale;
initial = min(shapeRectangleByCorners(grid,ankle-ankle_width,ankle+ankle_width), ...
    shapeRectangleByCorners(grid,toe-toe_width,toe+toe_width));
margin = nominal_contact_constraints(grid.xs, pendulum);
initial = max(initial, -margin);
assert(any(initial(:)<0), 'Grid misses the target interior; refine the grid without enlarging targets.');
time_s = 0:options.output_step_s:options.duration_s;
if time_s(end) < options.duration_s
    time_s(end+1) = options.duration_s;
end
scheme.grid = grid;
scheme.dynSys = pendulum;
scheme.accuracy = 'high';
scheme.uMode = 'min';
extra.obstacleFunction = margin;
extra.quiet = true;
extra.keepLast = false;
started = tic;
[value,tau] = HJIPDE_solve(initial,time_s/pendulum.omega,scheme,char(options.method),extra);
elapsed = toc(started);
export_hjb_policy(fullfile(output_dir,'policy.mat'),grid,value,tau,pendulum);
theta_rad = grid.vs{1};
velocity_rad_s = grid.vs{2}/pendulum.vscale;
projected_value = min(value(:,:,:,end),[],3);
save(fullfile(output_dir,'projection.mat'),'theta_rad','velocity_rad_s','projected_value','-v7');
summary = struct('profile',options.profile,'exo','none','direction',options.direction, ...
    'resolution',options.resolution,'duration_s',tau(end)*pendulum.omega, ...
    'output_step_s',options.output_step_s,'method',options.method,'wall_s',elapsed, ...
    'matlab_release',version('-release'),'gravity_m_s2',pendulum.g, ...
    'source_model',model_data.model,'target',target, ...
    'initial_target_nodes',nnz(initial<0),'final_reachable_nodes',nnz(value(:,:,:,end)<=0), ...
    'scope','Reconstructed nominal HJB baseline; convergence and published-area agreement unverified.');
file = fopen(fullfile(output_dir,'summary.json'),'w');
cleanup = onCleanup(@() fclose(file));
fprintf(file,'%s\n',jsonencode(summary,PrettyPrint=true));
end
