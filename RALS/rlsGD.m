function [yPrdtn,rals_training_err, rals_testing_err, U , V ] = rlsGD(Ytrn,par)

lambdaRLSGD= par.lambdaRLSGD;
k = par.k;
L = par.L; 
maxiter = par.maxiter;


rals_training_err = zeros(1,3);
rals_testing_err = zeros(1,3);

objgrad = @gradientRLSGD;
[n, m] = size(Ytrn);

v = randn(n*k+m*k,1);
[v,lp_idx,~] = steepestDescentRLSGD(v,Ytrn,objgrad,k,maxiter,lambdaRLSGD,L);

fprintf('\ntotal number of line search call in RALS = %d\n',lp_idx);

U = reshape(v(1:n*k),n,k);
V = reshape(v(n*k+1:n*k+m*k),m,k);

yPrdtn = round(U*V');
yPrdtn(yPrdtn<1) = 1;
yPrdtn(yPrdtn>L) = L;
end