function [y,U,V] = factorizeWeak(Ytrn,level,par)

[n,m] = size(Ytrn);
par.Y = Ytrn; 
p = par.p;
par.lambda = par.lambdaHMF(level);

clear Yrtn;
v = randn(n*p + m*p ,1);
%%
[v,numiter,ogcalls] = conjgrad(v,par);
fprintf('total conjugate gradient iteration in HMF = %d\n',numiter);
fprintf('total number of line search call in HMF = %d\n\n',ogcalls);

U = reshape(v(1:n*p),n,p);
V = reshape(v(n*p+1:n*p+m*p),m,p);
X = U * V';
y = mapRating(X,n,m);
end
%change log
%random U and V at every level 30/07/2014
%passing U and V obtained from previous level 30/07/2014
