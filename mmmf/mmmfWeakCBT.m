function [y,U,V,theta] = mmmfWeakCBT(weakTrn,par)

par.Y = weakTrn;
l = par.l;
[n, m] = size(weakTrn);
cb = par.cb;
[K1,K2] = size(cb);

v = randn(n*K1+m*K2+n*(l-1),1);
[v, numiter, ogcalls, J] = conjgrad(v,par);
fprintf('total conjugate gradient iteration in MMMF = %d\n',numiter);
fprintf('total number of line search call in MMMF = %d\n',ogcalls);
U = reshape(v(1:n*K1),n,K1);
V = reshape(v(n*K1+1:n*K1+m*K2),m,K2);
theta = reshape(v(n*K1+m*K2+1:n*K1+m*K2+n*(l-1)),n,l-1);
X = U*cb*V';
%clear v U V;
y = m3fSoftmax(X,theta);
end
