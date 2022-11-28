function [y,mmmf_training_err, mmmf_testing_err] = mmmfStrong(weakTrn,weakTst, strongTrn, strongTst,regvals)

%i = (1:40)./16;
%regvals = power(10,i);
%regvals = 27.3842;
%regvals = sqrt(sqrt(10)).^[8 7.5 7 6.5 6 5.5 5 4.5 4 3.5 3];

mmmf_training_err = zeros(size(regvals,2), 5);
mmmf_testing_err = zeros(size(regvals,2), 5);


objgrad = @m3fshc;
objgrad2 = @m3fshcFixedV;

[n,m] = size(weakTrn);
[n1,~] = size(strongTrn);

[p, ~, maxiter, l, ~, ~] = setParameter();

weakAll = weakTrn + weakTst;

tol = 1e-3;

v = randn(n*p+m*p+n*(l-1),1);
w = randn(n1*p+n1*(l-1),1);

for i=1:length(regvals)
    %% parameter weak
    parameter = {};
    parameter.lineSearchFun = @cgLineSearch;  parameter.c2 = 1e-2;
    parameter.objGrad = objgrad;              parameter.lambda = regvals(i);
    parameter.l = l;                          parameter.tol = tol;
    parameter.maxiter = maxiter;              parameter.Y = weakAll;
    %%
    [v, numiter, ogcalls] = conjgrad(v,parameter);
    %[v, numiter, ogcalls] = conjgrad(v,@cgLineSearch,{'c2',1e-2},objgrad,{weakAll,regvals(i),l,'verbose',0},'tol',tol,'maxiter',maxiter,'verbose',2);
    fprintf('\ntotal conjugate gradient iteration in MMMF for fixing V = %d\n',numiter);
    fprintf('total number of line search call in MMMF for fixing V= %d',ogcalls);
    %U = reshape(v(1:n*p),n,p);
    V = reshape(v(n*p+1:n*p+m*p),m,p);
    %%
    parameter.objGrad = objgrad2;                parameter.lambda = regvals(i);
    parameter.V = V;                             parameter.Y = strongTrn;
    %[w, numiter, ogcalls] = conjgrad(w,@cgLineSearch,{'c2',1e-2},objgrad2,{strongTrn,V,regvals(i),l,'verbose',0},'tol',tol,'maxiter',maxiter,'verbose',2);
    [w,numiter,ogcalls] = conjgrad(w,parameter);
    fprintf('\ntotal conjugate gradient iteration in MMMF for Strong Train = %d\n',numiter);
    fprintf('total number of line search call in MMMF for Strong Test = %d\n',ogcalls);
    U = reshape(w(1:n1*p),n1,p);
    theta = reshape(w(n1*p+1:n1*p+n1*(l-1)),n1,l-1);
    X = U*V';
    [y] = m3fSoftmax(X,theta);
    clear U V theta X;
    %%
    %fprintf('%d  p=%d  %.2e ZOE: %.2f %.4f  MAE: %.2f %.4f\n\n',i,p,regvals(i),zoe(y,strongTrn),zoe(y,strongTst),mae(y,strongTrn),mae(y,strongTst));
    mmmf_training_err(i,1) = zoe(y,strongTrn);
    mmmf_training_err(i,2) = mae(y,strongTrn);
    mmmf_training_err(i,3) = RMSE(y,strongTrn);
    mmmf_training_err(i,4) = averageMAE(y,strongTrn);
    mmmf_training_err(i,5) = averageRMSE(y,strongTrn);
    
    mmmf_testing_err(i,1) = zoe(y,strongTst);
    mmmf_testing_err(i,2) = mae(y,strongTst);
    mmmf_testing_err(i,3) = RMSE(y,strongTst);
    mmmf_testing_err(i,4) = averageMAE(y,strongTst);
    mmmf_testing_err(i,5) = averageRMSE(y,strongTst);
    
    
    %Y = Ytrain + Ytest;
    %mmmf_testing_err(4) = rre( y, Y);
end
%fprintf(1,'--------------------------------------------------\n\n');
end
