function [ alpha, flag ] = lineSearchRALSV(U,V, dV, obj_new, Ytrn, lambda)

maxiter = 6;
flag = 0;
i=1;
alpha = 1/36;
rho = 1/6;
obj_new1= inf;

while  pround(obj_new1,8) >= pround(obj_new,8)   && i <= maxiter 
    alpha = alpha * rho;
    tmpV = V - alpha .*dV;
    [obj_new1] = objectiveValueRALS(U,tmpV,Ytrn,lambda);
    i = i+1;
end

if i > maxiter
    flag=1;
end

%fprintf('\tline search Iteration: %d\n',i);
end