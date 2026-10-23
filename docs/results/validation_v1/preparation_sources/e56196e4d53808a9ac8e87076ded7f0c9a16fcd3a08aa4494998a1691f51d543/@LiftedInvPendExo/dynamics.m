function dx = dynamics(obj, t, x, u, d)

if obj.scaling
    l = 1;   
    m = 1;   
    g = 1; 
    bFric = obj.bFric*obj.tauscale*(1/obj.omega); % [Nm*s/rad] ; 
else
    l = obj.l;
    m = obj.m;
    g = obj.g;
    bFric = obj.bFric;
end

if iscell(x)
  dx = cell(obj.nx, 1);

  uExo = max(min((obj.Kd*(0-x{2}) + obj.Kp*((pi/2)-x{1})), obj.motorLim), -obj.motorLim) + obj.muGrav*m*g*l*saturatedCosVal(x{1},obj.m,obj.l, obj.motorLim, 0.01);
  f1 = x{2};
  f2 = -(g/l)*cos(x{1}) - (bFric/(m*l*l))*x{2} + (1/(m*l*l))*uExo;
  f3 = 0;

  
  g2 = 1/(m*l*l);
  g3 = 1;

   
  dx{1} = f1;
  dx{2} = f2 + g2.*x{3};
  dx{3} = f3 + g3.*u;
else
  dx = zeros(obj.nx, 1);

  uExo = max(min((obj.Kd*(0-x(2)) + obj.Kp*((pi/2)-x(1))), obj.motorLim), -obj.motorLim) + obj.muGrav*m*g*l*saturatedCosVal(x(1),obj.m,obj.l, obj.motorLim, 0.01);


  f1 = x(2);
  f2 = -(g/l)*cos(x(1)) - (bFric/(m*l*l))*x(2) + (1/(m*l*l))*uExo;
  f3 = 0;
 
  g2 = 1/(m*l*l);
  g3 = 1;
 
  dx(1) = f1;
  dx(2) = f2 + g2.*x(3);
  dx(3) = f3 + g3.*u;
end


end