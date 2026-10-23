function uOpt = optCtrl(obj, t, xs, deriv, uMode, ~)
if obj.scaling
    uMax = obj.uMax*obj.rtdscale;
    uMin = obj.uMin*obj.rtdscale;
else
    uMax = obj.uMax;
    uMin = obj.uMin;    
end
%% Input processing
if nargin < 5
  uMode = 'min';
end

%% Optimal control
if iscell(deriv)
    
 multiplier = deriv{3};
 
  uOpt = cell(obj.nu, 1);
  
  if strcmp(uMode, 'max')
    uOpt = (multiplier>=0)*uMax + (multiplier<0)* uMin;
    
  elseif strcmp(uMode, 'min')
    uOpt = (multiplier>=0)* uMin + (multiplier<0)*uMax;
    %size(uOpt)
  else
    error('Unknown uMode!')
  end  
  
else
  uOpt = zeros(obj.nu, 1);
  multiplier = deriv(3);
  if strcmp(uMode, 'max')
    uOpt = (multiplier>=0)*uMax + (multiplier<0)* uMin;
    
  elseif strcmp(uMode, 'min')
    uOpt = (multiplier>=0)* uMin + (multiplier<0)*uMax;
    
  else
    error('Unknown uMode!')
  end
end




end