function [obj,dV,lossobj,regobj] = gradientRALSV(U,V,Y,lambda)
 
  Ygt0 = Y>0;
  
  X = U*V';
 
  dV = lambda.*V; % [m,p]
  
  regobj = lambda.*(sum(U(:).^2)+sum(V(:).^2))./2; % [scalar]
  lossobj = sum(sum(Ygt0.*(Y - X ).^2));
  obj = regobj + lossobj; 

  dV = dV - 2.* (Ygt0.*(Y - X ))' * U;
  
end