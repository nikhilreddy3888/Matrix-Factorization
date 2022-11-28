function [ alpha,obj_new1,flag ] = lineSearchRLSGD(v, d, obj_new, Ytrn, lambda, k)

maxiter = 6;
flag = 0;
i=1;
alpha = 1/36;
rho = 1/6;
obj_new1= inf;
while  pround(obj_new1,8) >= pround(obj_new,8)   && i <= maxiter 
    alpha = alpha * rho;
    tmpv = v - alpha .*d;
    [obj_new1] = objectiveValueRLSGD(tmpv,Ytrn,lambda,k);
    i = i+1;
end

if i > maxiter
    flag=1;
end
%fprintf('\tline search Iteration: %d\n',i);
end