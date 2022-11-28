clear
addpath(genpath('.'));

i = (40:-1:1)./16;
regvals = power(10,i);
%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRun    = 3;
nRows   = 100;
nColumns= 100;
non0Per = 100;
tstPer  = 30;
k       = 100;
l       = 5; %Rating level
maxiter = 100;
tol     = 1e-3;
lambdaMMMF = regvals(21);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttlEvaluationMetrices = 3;

ResultTrnMMMF  = zeros(ttlEvaluationMetrices,nRun);
ResultTstMMMF  = zeros(ttlEvaluationMetrices,nRun);


filename = strcat( 'Result/resultFinal.txt');
fs = fopen(filename,'a');


for runNo = 1:nRun
    %% Data Generation
    Y = load('movielens.txt');
    %Y = generateData(nRows,nColumns,non0Per);
    %fprintf(fs,'\n\nrows:      %d\t column:       %d\t\t non0:  %d',size(Y,1),size(Y,2),non0Per);
    %% data pre-processing
    Y(sum(Y~=0,2)==0,:) = []; %code to delete user who has not given any rating
    Y = sparse(Y);
    [Ytrn,Ytst] = divideData(Y,tstPer);
    %[Ytrn,Ytst] = divideDataPMMMF(Y,tstPer);
    %[Ytrn, Ytst, ~, ~] = allBut1Division(Y, []);
    [n,m] = size(Ytrn);
    %fprintf(fs,'\nrows left: %d\t column left:  %d\n',n,m);
    L = full(max(max(Ytrn(:),Ytst(:))));
    minRating =full(min(min(Ytrn(Ytrn>0)), min(Ytst(Ytst>0))));
    %% Maximum Margin Matrix Factorization
    %
    par               = {};
    par.lineSearchFun = @cgLineSearch;
    par.c2            = 1e-2;
    par.objGrad       = @m3fshc;
    par.lambda        = lambdaMMMF;
    par.l             = L;
    par.tol           = tol;
    par.maxiter       = maxiter;
    par.p             = k;
    par.Y             = Ytrn;
    
    v0                = randn(n*k+m*k+n*(l-1),1); %U, V and Theta
    
    [v, numiter, ogcalls, J] = conjgrad(v0,par);
    
    U                 = reshape(v(1:n*k),n,k);
    V                 = reshape(v(n*k+1:n*k+m*k),m,k);
    theta             = reshape(v(n*k+m*k+1:n*k+m*k+n*(l-1)),n,l-1);
    X                 = U*V';
    yMMMFPre          = m3fSoftmax(X,theta);

    ResultTrnMMMF(:,runNo) = EvaluationAll(yMMMFPre, Ytrn);
    ResultTstMMMF(:,runNo) = EvaluationAll(yMMMFPre, Ytst);
    %}
end


ResultTrnMMMFAvg = mean(ResultTrnMMMF,2);
ResultTstMMMFAvg = mean(ResultTstMMMF,2);


fprintf(fs,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTrnMMMFAvg(1,1),ResultTrnMMMFAvg(2,1),ResultTrnMMMFAvg(3,1));
fprintf(fs,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTstMMMFAvg(1,1),ResultTstMMMFAvg(2,1),ResultTstMMMFAvg(3,1));
