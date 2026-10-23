classdef PaperExoPendulum < LiftedInvPendExo
    %PAPEREXOPENDULUM Reconciled paper GC/restoring-PD dynamics for helperOC.
    % Inherits the authors' anthropometry, scaling and optimal torque-rate bounds.
    % Controller gains and motor limits remain SI; scaling occurs after dynamics.
    properties
        assistance
    end
    methods
        function obj = PaperExoPendulum(model, assistance)
            assistance = string(assistance);
            assert(ismember(assistance,["none","gc","pd"]),'Unknown assistance mode.');
            params.alphaMT = 1;
            params.alphaRTD = 1;
            params.exoParams = struct('Kp',0,'Kd',0,'muGrav',0,'motorLim',0);
            obj@LiftedInvPendExo([pi/2;0;0],model,params);
            obj.assistance = string(assistance);
            if assistance ~= "none"
                obj.motorLim = 50;
            end
            if assistance == "gc"
                obj.muGrav = 1;
            elseif assistance == "pd"
                obj.Kp = obj.m*obj.g*obj.l;
                obj.Kd = 0.3*sqrt(obj.m*obj.l^2*obj.Kp);
            end
        end
        function torque = exoTorqueSI(obj, theta, velocity)
            % Bilateral torque [Nm], absolute theta [rad], velocity [rad/s].
            raw = obj.muGrav*obj.m*obj.g*obj.l*cos(theta) ...
                + obj.Kp*(pi/2-theta) - obj.Kd*velocity;
            torque = max(-obj.motorLim,min(obj.motorLim,raw));
        end
        function dx = dynamics(obj, ~, x, u, ~)
            % State and rate use the inherited nondimensional scales if enabled.
            cell_input = iscell(x);
            if ~cell_input
                x = num2cell(x);
            end
            velocity = x{2};
            human_torque = x{3};
            if obj.scaling
                velocity = velocity/obj.vscale;
                human_torque = human_torque/obj.tauscale;
            end
            acceleration = (-obj.m*obj.g*obj.l*cos(x{1}) ...
                - obj.bFric*velocity + human_torque ...
                + obj.exoTorqueSI(x{1},velocity))/(obj.m*obj.l^2);
            dx = {velocity;acceleration;u};
            if obj.scaling
                dx{1} = dx{1}*obj.vscale;
                dx{2} = dx{2}*obj.omega*obj.vscale;
            end
            if ~cell_input
                dx = cell2mat(dx);
            end
        end
    end
end
