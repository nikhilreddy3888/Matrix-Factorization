function [results] = hmfTstLambda(Ytrn, Ytst, par)

[n,m] = size(Ytrn);
par.Y = Ytrn; 
p = par.p;
levelNo = par.levelNo;


RcapTrn = zeros(n, m);
RcapTst = zeros(n, m);
        
RcapTrn( (Ytrn <= levelNo) & ( Ytrn ~= 0 ) ) = -1;
RcapTst( (Ytst <= levelNo) & ( Ytst ~= 0 ) ) = -1;
        
RcapTrn( (Ytrn > levelNo) & ( Ytrn ~= 0 ) ) = 1;
RcapTst( (Ytst > levelNo) & ( Ytst ~= 0 ) ) = 1;
        
RcapTrn = sparse(RcapTrn);
RcapTst = sparse(RcapTst);

par.Y = RcapTrn;

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

results = EvaluationAll(y, RcapTrn + RcapTst, 0);
   
clear RcapTrn RcapTst;
end


        
    
    
    