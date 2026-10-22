function summary = solve_nominal_hjb(author_root, dependency_root, output_dir, options)
%SOLVE_NOMINAL_HJB Original nominal model/targets with source-domain defaults.
% Preserves the 51x225x51, 0.02 rad/s buffer and 100 ms output baseline.
% solve_paper_hjb owns the shared solve/export path and assisted variants.
arguments
    author_root (1,1) string
    dependency_root (1,1) string
    output_dir (1,1) string
    options.profile (1,1) string {mustBeMember(options.profile,["YF","OF","YM","OM","wOF"])} = "OF"
    options.direction (1,1) string {mustBeMember(options.direction,["forward","backward"])} = "forward"
    options.resolution (3,1) double {mustBeInteger,mustBePositive} = [51;225;51]
    options.duration_s (1,1) double {mustBePositive,mustBeFinite} = 1.4
    options.output_step_s (1,1) double {mustBePositive,mustBeFinite} = 0.1
    options.velocity_buffer_rad_s (1,1) double {mustBePositive,mustBeFinite} = 0.02
    options.quiet (1,1) logical = false
    options.method (1,1) string {mustBeMember(options.method,["none","minVWithV0"])} = "none"
end
name_values = namedargs2cell(options);
summary = solve_paper_hjb(author_root,dependency_root,output_dir,name_values{:});
end
