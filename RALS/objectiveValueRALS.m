function [obj] = objectiveValueRALS(U,V, Y, lambda)

 Ygt0 = Y>0;
  
  X = U*V';
  
  regobj = lambda.*(sum(U(:).^2)+sum(V(:).^2))./2; % [scalar]
  lossobj = sum(sum(Ygt0.*(Y - X ).^2));
  obj = regobj + lossobj; 

end