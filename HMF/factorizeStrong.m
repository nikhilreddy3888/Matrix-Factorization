function [y] = factorizeStrong(weakAllCap, strongCapTrn, level,flagSdCg,lambda)
%
gradfun = @m3fshcBinary; %gradient function
gradfun1 = @m3fshcBinaryFixedV;

[p, alpha, maxiter, ~, ~, ~] = setParameter();%p:-number of latent factor; alpha:-step size
[n, m] = size(weakAllCap);
v = randn(n*p + m*p ,1);

[n1, m1] = size(strongCapTrn);
w = randn(n1*p, 1);
tol = 1e-3;

if flagSdCg == 1
    maxiter= 200;
end

%{
%steepest Descent code need to be modified so that it can work for at the
time of strong call
if flagSdCg == 1
    [v,numiter] = steepestDescent(v,weakAllCap,gradfun,p,maxiter,lambda(level),alpha);
    fprintf('total Steepest Descent gradient iteration = %d\n\n',numiter);
    V = reshape(v(n*p+1:n*p+m*p),m,p);
    clear v;
    %need to write steepestDescentStrong module
    [w,numiter] = steepestDescent(w,strongCapTrn,V,gradfun1,p,maxiter,lambda(level));
    fprintf('total conjugate gradient iteration in Strong Train in HMF = %d\n',numiter);
    
end
%}

if flagSdCg == 2
    %%  Parameter Weak
    parameter = {};
    parameter.lineSearchFun = @cgLineSearch;    parameter.c2 = 1e-2;
    parameter.objGrad = gradfun;                parameter.lambda = lambda(level);
    parameter.tol = tol;                        parameter.maxiter = maxiter;
    parameter.Y = weakAllCap;                   parameter.p = p;
    %%
    [v,numiter,ogcalls] = conjgrad(v,parameter);
    %[v,numiter,ogcalls] = cgWeak(v,weakAllCap,gradfun,p,maxiter,lambda(level));
    fprintf('total conjugate gradient iteration in HMF for Fixing V at level %d = %d\n',level,numiter);
    fprintf('total number of line search call in HMF for Fixing V at level %d = %d\n',level,ogcalls);
    V = reshape(v(n*p+1:n*p+m*p),m,p);
    %clear v;
    %% parameter Strong
    parameter.objGrad = gradfun1;                parameter.lambda = lambda(level);
    parameter.V = V;                             parameter.Y = strongCapTrn;
    %%
    [w,numiter,ogcalls] = conjgrad(w,parameter);
    %[w,numiter,ogcalls] = cgStrong(w,strongCapTrn,V,gradfun1,p,maxiter,lambda(level));
    fprintf('total conjugate gradient iteration in Strong Train in HMF at level %d = %d\n',level,numiter);
    fprintf('total number of line search call in Strong Train in HMF at level %d = %d\n',level,ogcalls);    
end
U = reshape(w(1:n1*p),n1,p);
%V = reshape(v(n*p+1:n*p+m*p),m,p);
X = U * V';
y = mapRating(X,n1,m1);
clear U V X;
end

%change log
% passing U, V and U',V' obtained from previous level
