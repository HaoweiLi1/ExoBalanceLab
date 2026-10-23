function summary = solve_paper_hjb(author_root, dependency_root, output_dir, options)
%SOLVE_PAPER_HJB Nominal source or reconciled GC/PD HJB with C1/C3-C5.
% Nominal mode retains the authors' dynamics and stored box targets. Assisted
% modes use SI-consistent dynamics and recomputed Appendix II target boxes.
arguments
    author_root (1,1) string
    dependency_root (1,1) string
    output_dir (1,1) string
    options.profile (1,1) string {mustBeMember(options.profile,["YF","OF","YM","OM","wOF"])} = "OF"
    options.exo (1,1) string {mustBeMember(options.exo,["none","gc","pd"])} = "none"
    options.direction (1,1) string {mustBeMember(options.direction,["forward","backward"])} = "forward"
    options.resolution (3,1) double {mustBeInteger,mustBePositive} = [51;269;51]
    options.duration_s (1,1) double {mustBePositive,mustBeFinite} = 1.4
    options.output_step_s (1,1) double {mustBePositive,mustBeFinite} = 0.01
    options.velocity_buffer_rad_s (1,1) double {mustBePositive,mustBeFinite} = 0.5
    options.quiet (1,1) logical = false
    options.prepare_only (1,1) logical = false
    options.align_zero (1,1) logical = false
    options.method (1,1) string {mustBeMember(options.method,["none","minVWithV0"])} = "none"
    options.mt_scale (1,1) double {mustBePositive,mustBeFinite} = 1
    options.mrtd_scale (1,1) double {mustBePositive,mustBeFinite} = 1
    options.target_definition (1,1) string {mustBeMember(options.target_definition, ...
        ["legacy","recomputed","source_entry","common_ankle"])} = "legacy"
end
assert(options.mt_scale<=1 && options.mrtd_scale<=1,'Strength scales must not exceed one.');
assert(options.target_definition~="legacy" || ...
    (options.mt_scale==1 && options.mrtd_scale==1), ...
    'Legacy stored targets are only defined at nominal strength; select an explicit target definition.');
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
if options.exo ~= "none" || options.target_definition ~= "legacy"
    addpath(genpath(fullfile(dependency_root,'YALMIP')));
    addpath(genpath(fullfile(dependency_root,'sedumi')));
end
% MATLAB discovers the external dynamics/optCtrl methods only in an @ folder.
% Stage unchanged source into the run, leaving the authors' checkout untouched.
mkdir(output_dir);
class_root = fullfile(output_dir,'source');
mkdir(class_root);
copyfile(fullfile(author_root,'Dynamics','LiftedInvPendExo'), ...
    fullfile(class_root,'@LiftedInvPendExo'));
addpath(class_root);
copyfile(model_file, fullfile(output_dir,'source_model.mat'));
if options.exo == "none" && options.target_definition == "legacy"
    target_data = load(target_file, 'problemParams');
    copyfile(target_file, fullfile(output_dir,'source_target.mat'));
    pendulum = LiftedInvPendExo([pi/2;0;0], model_data.model, target_data.problemParams);
    target = target_data.problemParams.targetSets;
else
    pendulum = PaperExoPendulum(model_data.model,options.exo,options.mt_scale,options.mrtd_scale);
    [target,target_diagnostics] = paper_target_boxes(pendulum);
    if options.target_definition == "source_entry"
        % Explicit overrides in the authors' computeAnkleExoBRS.m entry point.
        % Both velocity widths are changed; these boxes are finite-time goals.
        target.thetaDotAnkRange = 0.01;
        target.thetaDotToeRange = 0.01;
        target.torqueAnkRange = 1;
    elseif options.target_definition == "common_ankle"
        % An identical SI goal across modes/strengths, for a separate sensitivity
        % experiment. Duplicate the ankle box to preserve the exported schema.
        target.ankleEq = [pi/2;0;0];
        target.toeEq = target.ankleEq;
        target.thetaAnkRange = 0.005;
        target.thetaToeRange = 0.005;
        target.thetaDotAnkRange = 0.01;
        target.thetaDotToeRange = 0.01;
        target.torqueAnkRange = 1;
        target.torqueToeRange = 1;
    end
    save(fullfile(output_dir,'computed_target.mat'),'target','target_diagnostics');
end
for file_name = ["solve_paper_hjb.m","PaperExoPendulum.m","paper_target_boxes.m", ...
        "paper_contact_constraints.m","nominal_contact_constraints.m","export_hjb_policy.m"]
    copyfile(fullfile(fileparts(mfilename('fullpath')),file_name),fullfile(output_dir,'source',file_name));
end
if options.direction == "forward"
    lower = [0.65;-options.velocity_buffer_rad_s*pendulum.vscale;pendulum.torqueMin*pendulum.tauscale];
    upper = [1.8;2.85*pendulum.vscale;pendulum.torqueMax*pendulum.tauscale];
else
    lower = [1.51;-2.85*pendulum.vscale;pendulum.torqueMin*pendulum.tauscale];
    upper = [2.56;options.velocity_buffer_rad_s*pendulum.vscale;pendulum.torqueMax*pendulum.tauscale];
end
resolution = options.resolution;
if options.exo ~= "none" || options.align_zero
    % Small quiet-standing targets require zero speed/torque to be sampled.
    % Keep uniform spacing, extending each domain by at most one cell per side.
    % C1/contact obstacles still enforce the original physical bounds.
    for axis = [2,3]
        spacing = (upper(axis)-lower(axis))/(resolution(axis)-1);
        indices = [floor(lower(axis)/spacing+1e-12),ceil(upper(axis)/spacing-1e-12)];
        lower(axis) = indices(1)*spacing;
        upper(axis) = indices(2)*spacing;
        resolution(axis) = diff(indices)+1;
    end
end
grid = createGrid(lower, upper, resolution);
ankle_width = [target.thetaAnkRange;target.thetaDotAnkRange*pendulum.vscale; ...
    target.torqueAnkRange*pendulum.tauscale];
toe_width = [target.thetaToeRange;target.thetaDotToeRange*pendulum.vscale; ...
    target.torqueToeRange*pendulum.tauscale];
scale = [1;pendulum.vscale;pendulum.tauscale];
ankle = target.ankleEq(:).*scale;
toe = target.toeEq(:).*scale;
ankle_value = shapeRectangleByCorners(grid,ankle-ankle_width,ankle+ankle_width);
toe_value = shapeRectangleByCorners(grid,toe-toe_width,toe+toe_width);
if options.exo == "none"
    margin = nominal_contact_constraints(grid.xs,pendulum);
else
    device = pendulum.exoTorqueSI(grid.xs{1},grid.xs{2}/pendulum.vscale);
    margin = paper_contact_constraints(grid.xs,pendulum,device);
end
ankle_nodes = nnz(max(ankle_value,-margin)<0);
toe_nodes = nnz(max(toe_value,-margin)<0);
assert(ankle_nodes>0 && toe_nodes>0, ...
    'Grid misses a target interior; refine the grid without enlarging targets.');
initial = max(min(ankle_value,toe_value),-margin);
time_s = 0:options.output_step_s:options.duration_s;
if time_s(end) < options.duration_s
    time_s(end+1) = options.duration_s;
end
scheme.grid = grid;
scheme.dynSys = pendulum;
scheme.accuracy = 'high';
scheme.uMode = 'min';
extra.obstacleFunction = margin;
extra.quiet = options.quiet;
extra.keepLast = false;
contract.model = 'paper';
contract.exo = options.exo;
contract.profile = struct('mass_kg',pendulum.m+pendulum.mf,'height_m',pendulum.l/0.575, ...
    'torque_min_nm',pendulum.torqueMin,'torque_max_nm',pendulum.torqueMax, ...
    'rate_min_nm_s',pendulum.uMin,'rate_max_nm_s',pendulum.uMax, ...
    'damping_nm_s',pendulum.bFric,'friction',1,'gravity_m_s2',pendulum.g);
theta_rad = grid.vs{1};
velocity_rad_s = grid.vs{2}/pendulum.vscale;
summary = struct('profile',options.profile,'exo',options.exo,'direction',options.direction, ...
    'resolution',resolution,'requested_resolution',options.resolution, ...
    'duration_s',time_s(end), ...
    'output_step_s',options.output_step_s,'method',options.method, ...
    'velocity_buffer_rad_s',options.velocity_buffer_rad_s, ...
    'aligned_zero',options.exo ~= "none" || options.align_zero, ...
    'matlab_release',version('-release'),'gravity_m_s2',pendulum.g, ...
    'source_model',model_data.model,'target',target, ...
    'mt_scale',options.mt_scale,'mrtd_scale',options.mrtd_scale, ...
    'target_definition',options.target_definition,'effective_profile',contract.profile, ...
    'target_semantics','Finite-horizon terminal goal intersected with C1/C3-C5; box invariance is not asserted.', ...
    'ankle_target_nodes',ankle_nodes,'toe_target_nodes',toe_nodes, ...
    'initial_target_nodes',nnz(initial<0), ...
    'scope','Source nominal / reconciled assisted HJB; convergence and published-area agreement unverified.');
if options.prepare_only
    % Export the actual MATLAB fields for an independently checked GPU integrator.
    % No HJB equation is solved by this preparation step.
    summary.stage = 'prepared';
    human_torque_nm = grid.vs{3}/pendulum.tauscale;
    time_to_go_s = time_s(:);
    state_scale = [1;pendulum.vscale;pendulum.tauscale];
    time_scale_s = pendulum.omega;
    rate_bounds = [pendulum.uMin;pendulum.uMax]*pendulum.rtdscale;
    dx = pendulum.dynamics(0,grid.xs,0,[]);
    drift_theta = dx{1};
    drift_velocity = dx{2};
    initial_value = initial;
    obstacle_value = -margin;
    problem_json = jsonencode(struct('summary',summary,'contract',contract));
    save(fullfile(output_dir,'problem.mat'),'theta_rad','velocity_rad_s','human_torque_nm', ...
        'time_to_go_s','state_scale','time_scale_s','rate_bounds','drift_theta', ...
        'drift_velocity','initial_value','obstacle_value','problem_json','-v7.3');
else
    started = tic;
    [value,tau] = HJIPDE_solve(initial,time_s/pendulum.omega,scheme,char(options.method),extra);
    summary.wall_s = toc(started);
    summary.duration_s = tau(end)*pendulum.omega;
    summary.final_reachable_nodes = nnz(value(:,:,:,end)<=0);
    summary.backend = 'MATLAB/helperOC';
    summary.stage = 'solved';
    export_hjb_policy(fullfile(output_dir,'policy.mat'),grid,value,tau,pendulum,contract);
    projected_value = min(value(:,:,:,end),[],3);
    save(fullfile(output_dir,'projection.mat'),'theta_rad','velocity_rad_s','projected_value','-v7');
end
file = fopen(fullfile(output_dir,'summary.json'),'w');
cleanup = onCleanup(@() fclose(file));
fprintf(file,'%s\n',jsonencode(summary,PrettyPrint=true));
end
