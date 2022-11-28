function [obj,dU] = gradientRALSU(U,V,Y,lambda)
 

  clear v;

  Ygt0 = Y>0;
  
  X = U*V';
 
  dU = lambda.*U; % [n,p]

  
  regobj = lambda.*(sum(U(:).^2)+sum(V(:).^2))./2; % [scalar]
  lossobj = sum(sum(Ygt0.*(Y - X ).^2));
  obj = regobj + lossobj; 
  
  dU = dU - 2.* (Ygt0.*(Y - X ))*V;
end