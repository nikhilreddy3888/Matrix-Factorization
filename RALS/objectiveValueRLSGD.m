function [obj] = objectiveValueRLSGD(v, Y, lambda, k)

  [n,m] = size(Y);
  
  U = reshape(v(1:n*k),n,k);
  V = reshape(v(n*k+1:n*k+m*k),m,k);
 
  clear v;

  Ygt0 = Y>0;
  
  X = U*V';
  
  regobj = lambda.*(sum(U(:).^2)+sum(V(:).^2))./2; % [scalar]
  lossobj = sum(sum(Ygt0.*(Y - X ).^2));
  obj = regobj + lossobj; 

end
%{
function [ret] = h(z)
  ret = ((z-1).^2).*(z<1);
%}
