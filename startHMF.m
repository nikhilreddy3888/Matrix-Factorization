clear
addpath(genpath('.'));

i = (40:-1:1)./16;
regvals = power(10,i);

%% MMMF EachMovie regularization value in case of 
%lambdaM3f = regvals(12); 
%% MMMF Movielens 1m regularization value
%lambdaM3f = regvals(18); %k =100
%lambdaM3f = regvals(18); %k =80
%lambdaM3f = regvals(20); %k =60
%lambdaM3f = regvals(18); %k =40 
%lambdaM3f = regvals(22); %k =20 
%% MMMF Movielens 100K regularization value
%lambdaM3f = regvals(25);  %k =100 
%lambdaM3f = regvals(18); %k =80
%lambdaM3f = regvals(20); %k =60
%lambdaM3f = regvals(18); %k =40 
%lambdaM3f = regvals(22); %k =20 
%% HMF Eachmovie regularization value
%lambdaHmf = [regvals(20), regvals(20), regvals(15), regvals(15),regvals(17)]; 
%lambdaHmf = [ regvals(21), regvals(20), regvals(17), regvals(16),regvals(19)];
%% HMF Movielens 1m regularization value  
%lambdaHmf = [ regvals(26), regvals(22), regvals(19), regvals(19)]; %k =100
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =80
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =60
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =40
%lambdaHmf = [ regvals(24), regvals(22), regvals(22), regvals(24)]; %k =20
%% HMF Movielens 100k regularization value
%lambdaHmf = [ regvals(29), regvals(28), regvals(25), regvals(27)]; %k =100
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =80
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =60
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =40
%lambdaHmf = [ regvals(24), regvals(22), regvals(22), regvals(24)]; %k =20
%% HMF dummy regularization value
%lambdaHmf = [ 0.5, 0.5, 0.5, 0.5];
%lambdaHmf = [ 1 1 1 1];
%% %%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRun    = 1;
nRows   = 100;
nColumns= 100;
non0Per = 30;
tstPer  = 30;
k       = 100;
l       = 5;
maxiter = 100;
tol     = 1e-3;
lambdaHMF = [regvals(29), regvals(28), regvals(25), regvals(27)]; %level-wise lambda
lambdaMMMF = regvals(21);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ttlEvaluationMetrices = 3;
ResultTrnHMF = zeros(ttlEvaluationMetrices,nRun);
ResultTrnMMMF  = zeros(ttlEvaluationMetrices,nRun);

ResultTstHMF = zeros(ttlEvaluationMetrices,nRun);
ResultTstMMMF  = zeros(ttlEvaluationMetrices,nRun);


filename = strcat( 'Result/resultFinal.txt');
fs = fopen(filename,'a');


parfor runNo = 1:nRun
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
    %% Hierarchical Maximum Margin Matrix Factorization
    %
    par = {};
    par.lineSearchFun = @cgLineSearch;    par.c2 = 1e-2;
    par.objGrad = @m3fshcBinary;          par.lambdaHMF = lambdaHMF ;
    par.tol = tol;                        par.maxiter = maxiter;
    par.p = k;                            par.ratingLevel = L;
    par.increment = 1;                    par.minRating = minRating;
    
    
    [RFinal,ULevelWise,VLevelWise,mf_Lvl_trn_err,hmf_Lvl_tst_err] = ...
        hmfTstWeak(Ytrn,Ytst, par);
    
    ResultTrnHMF(:,runNo) = EvaluationAll(RFinal, Ytrn);
    ResultTstHMF(:,runNo) = EvaluationAll(RFinal, Ytst);
    %}
    %% Maximum Margin Matrix Factorization
    %
    par = {};
    par.lineSearchFun = @cgLineSearch;  par.c2 = 1e-2;
    par.objGrad = @m3fshc;              par.lambda = lambdaMMMF;
    par.l = L;                          par.tol = tol;
    par.maxiter = maxiter;              par.p = k;
    
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


ResultTrnHMFAvg = mean(ResultTrnHMF,2);
ResultTrnMMMFAvg = mean(ResultTrnMMMF,2);

ResultTstHMFAvg = mean(ResultTstHMF,2);
ResultTstMMMFAvg = mean(ResultTstMMMF,2);


fprintf(fs,'\n\nHMF Training Error:\t\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTrnHMFAvg(1,1),ResultTrnHMFAvg(2,1),ResultTrnHMFAvg(3,1));
fprintf(fs,'\nHMF Testing Error:\t\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4ff',...
    ResultTstHMFAvg(1,1),ResultTstHMFAvg(2,1),ResultTstHMFAvg(3,1));


fprintf(fs,'\n\nMMMF-CG Training Error:     ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTrnMMMFAvg(1,1),ResultTrnMMMFAvg(2,1),ResultTrnMMMFAvg(3,1));
fprintf(fs,'\nMMMF-CG Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f',...
    ResultTstMMMFAvg(1,1),ResultTstMMMFAvg(2,1),ResultTstMMMFAvg(3,1));
