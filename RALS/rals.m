function [yPrdtn, U , V ] = rals(Ytrn,par)

lambdaRALS = par.lambdaRALS;
k = par.k;
L = par.L; 
maxiter = par.maxiter;


objgradU = @gradientRALSU;
objgradV = @gradientRALSV;

[n, m] = size(Ytrn);

U = randn(n,k);
V = randn(m,k);

[U,V,lp_idx] = steepestDescentRALS(U,V,Ytrn,objgradU,objgradV,maxiter,lambdaRALS);

fprintf('\ntotal number of line search call in RALS = %d\n',lp_idx);



yPrdtn= round(U*V');
yPrdtn(yPrdtn<1) = 1;
yPrdtn(yPrdtn>L) = L;
end