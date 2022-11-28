function [ alpha,obj_new1,flag ] = lineSearch(v, d, obj_new, Rtrain, lambda, p)

maxiter = 10;
flag = 0;
i=1;
[n,m] = size(Rtrain);
alpha = 1/6;
tmpv = v;
rho = 1/6;
tmpv(1:n*p+m*p) = v(1:n*p+m*p) - alpha .*d(1:n*p+m*p);
[obj_new1, ~] = m3fshcBinary(v, Rtrain, lambda, p);

while  obj_new1 >= obj_new   && i <= maxiter 
    alpha = alpha * rho;
    tmpv(1:n*p+m*p) = v(1:n*p+m*p) - alpha .*d(1:n*p+m*p);
    [obj_new1, ~] = m3fshcBinary(tmpv,Rtrain,lambda,p);
    i = i+1;
end

if i > maxiter
    flag=1;
end
%fprintf('\tline search Iteration: %d',i);
end