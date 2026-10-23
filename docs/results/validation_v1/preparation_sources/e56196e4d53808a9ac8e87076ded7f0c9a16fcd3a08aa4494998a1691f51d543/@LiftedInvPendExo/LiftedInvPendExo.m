classdef LiftedInvPendExo < DynSys
  % Constructs a lifted inverted pendulum with exo dynamics
  % Scaling constants are class members, but scaling happens outside of the
  % class.
    properties
    % Input bounds
    uMin
    uMax

    

    % IP model Params
    l
    m
    g
    bFric
    torqueMin
    torqueMax

    % Foot params
    ank
    b
    c
    lf
    mf

    % Controller params
    Kd
    Kp
    muGrav
    motorLim

    % Nondimensionalizing params
    scaling
    omega 
    mscale
    vscale
    tauscale
    rtdscale
    

  end
  
  methods
    function obj = LiftedInvPendExo(x, modelParams, problemParams)
      
      if numel(x) ~= 3
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.uMin = modelParams.uMin*problemParams.alphaRTD;
      obj.uMax = modelParams.uMax*problemParams.alphaRTD;
      
      obj.pdim = 1;
      
      obj.nx = 3;
      obj.nu = 1;

      obj.torqueMin = modelParams.tauMin*problemParams.alphaMT;
      obj.torqueMax = modelParams.tauMax*problemParams.alphaMT;

            % Rescale again
      obj.g = 9.8; % [m/s^2]    acceleration of gravity
      obj.bFric = 0.001; % [s*Nm/rad] friction coefficient

      [obj.l, obj.lf, obj.mf, obj.ank, obj.m, obj.b, obj.c] = ...
          proportionallyEstimatedParams(modelParams.m, modelParams.h);

      
      % Controller parameters
      obj.Kp = problemParams.exoParams.Kp;
      obj.Kd = problemParams.exoParams.Kd;
      obj.muGrav = problemParams.exoParams.muGrav;
      obj.motorLim = problemParams.exoParams.motorLim;

      % Nondimensionalizing parameters
      obj.scaling = true;
      obj.mscale = 1;
      obj.omega = sqrt(obj.l/9.8);
      obj.vscale = 1*obj.omega;
      obj.tauscale = obj.mscale*(obj.omega^2)*(1/(obj.m*obj.l*obj.l));
      obj.rtdscale = obj.tauscale*obj.omega;


    end

  end % end methods
end % end classdef
