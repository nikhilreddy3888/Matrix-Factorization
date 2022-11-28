clear;
addpath(genpath('.'));
%load('data\Movielens-1m-mine-curr');
%load('data\EachMovie');

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
lambdaM3f = regvals(25);  %k =100 
lambdaM3f = 1/1.6;
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
lambdaHmf = [ regvals(29), regvals(28), regvals(25), regvals(27)]; %k =100
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =80
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =60
%lambdaHmf = [ regvals(26), regvals(21), regvals(19), regvals(20)]; %k =40
%lambdaHmf = [ regvals(24), regvals(22), regvals(22), regvals(24)]; %k =20
%% HMF dummy regularization value
%lambdaHmf = [ 0.5, 0.5, 0.5, 0.5];
%lambdaHmf = [ 1 1 1 1];
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%nRun = size(weaktest, 2);
nRun    = 1;
nRows    = 50;
nColumns= 50;
nRowInWeak = 40;
non0Per = 30;
tstPer  = 20;
k       = 100;
L       = 5;
maxiter = 200;
tol     = 1e-3;
lambdaHMF = lambdaHmf;
lambdaMMMF = lambdaM3f;

topk   = 10;
cutoff = 10;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ttlEvaluationMetrices = 5;
ResultTrnHMF = zeros(ttlEvaluationMetrices,nRun);
ResultTrnMMMF  = zeros(ttlEvaluationMetrices,nRun);

ResultTstHMF = zeros(ttlEvaluationMetrices,nRun);
ResultTstMMMF  = zeros(ttlEvaluationMetrices,nRun);


filename = strcat( 'Result/resultFinal.txt');
fs = fopen(filename,'a');

for runNo=1:nRun
%% Data Generation
    Y = load('movielens.txt');
    %Y = generateData(nRows,nColumns,non0Per);    
    fprintf(fs,'\n\nrows:      %d\t column:       %d\t\t non0:  %d',size(Y,1),size(Y,2),non0Per);    %
    nRowInWeak = size(Y,1);
    [weakR,strongR] = divideIntoWeakStrong(Y, nRowInWeak);
    %[weakTrn, weakTst, strongTrn, strongTst] = allBut1Division(weakR, strongR); % This is not weak division code
    
    %% tmporary code
    noRating = 20;
    noOfRatinginForTrain = 10;
    tmpIdx   = sum(weakR~=0,2)>=noRating; weakR = weakR(tmpIdx,:);
    [weakTrn, weakTst] = divideData2(weakR, noOfRatinginForTrain);
    
    
    %%
    
    %weakTrn = [3 0 0 5 2 0 0;5 4 0 1 5 3 4; 1 0 4 0 3 1 0; 5 4 0 0 0 0 1; 0 3 2 0 5 2 0];
    %weakTrn = randi([0,5],5,1000);
    %weakTst = zeros(5,1000);
    %[weakTrn,weakTst,~,~] = allBut1Division(weakR, strongR);
    %weakTrn = sparse(weakTrn);
    %weakTst = sparse(weakTst);
    %}
    %
    %weakTrn = weaktrain{runNo};
    %weakTst = weaktest{runNo};
    %}
    
    L = full(max(max(weakTrn(:),weakTst(:))));
    minRating =full(min(min(weakTrn(weakTrn>0)), min(weakTst(weakTst>0))));

    %%  Maximum Margin Matrix Factorization
    %
    par = {};
    par.lineSearchFun = @cgLineSearch;  par.c2 = 1e-2;
    par.objGrad = @m3fshc;              par.lambda = lambdaMMMF;
    par.l = L;                          par.tol = tol;
    par.maxiter = maxiter;              par.p = k;

    [yMMMF, U , V ,mmmfTheta] = mmmfWeak(weakTrn, par);
    ResultTrnMMMF(:,runNo) = EvaluationAll(yMMMF, weakTrn, topk, cutoff);
    ResultTstMMMF(:,runNo) = EvaluationAll(yMMMF, weakTst, topk, cutoff);
    %}   
    %% Hierarchical Maximum Margin Matrix Factorization
    %
    par = {};
    par.lineSearchFun = @cgLineSearch;    par.c2 = 1e-2;
    par.objGrad = @m3fshcBinary;          par.lambdaHMF = lambdaHMF ;
    par.tol = tol;                        par.maxiter = maxiter;
    par.p = k;                            par.ratingLevel = L;
    par.increment = 1;                    par.minRating = minRating;
    
    
    [RFinal,ULevelWise,VLevelWise,mf_Lvl_trn_err,hmf_Lvl_tst_err] = ...
        hmfTstWeak(weakTrn,weakTst, par);
    
    ResultTrnHMF(:,runNo) = EvaluationAll(RFinal, weakTrn, topk, cutoff);
    ResultTstHMF(:,runNo) = EvaluationAll(RFinal, weakTst, topk, cutoff);
    %}
    %}
end

ResultTrnHMFAvg = mean(ResultTrnHMF,2);
ResultTrnMMMFAvg = mean(ResultTrnMMMF,2);

ResultTstHMFAvg = mean(ResultTstHMF,2);
ResultTstMMMFAvg = mean(ResultTstMMMF,2);

fprintf(fs,'\n\nHMF-Weak Training Error:\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\t\tRRMSE = %.4f',...
    ResultTrnHMFAvg(1,1),ResultTrnHMFAvg(2,1),ResultTrnHMFAvg(3,1),ResultTrnHMFAvg(4,1));
fprintf(fs,'\nHMF-Weak Testing Error:\t\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\t\tRRMSE = %.4f',...
    ResultTstHMFAvg(1,1),ResultTstHMFAvg(2,1),ResultTstHMFAvg(3,1),ResultTstHMFAvg(4,1));


fprintf(fs,'\n\nMMMF-CG-Weak Training Error:ZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\t\tRRMSE = %.4f',...
    ResultTrnMMMFAvg(1,1),ResultTrnMMMFAvg(2,1),ResultTrnMMMFAvg(3,1),ResultTrnMMMFAvg(4,1));
fprintf(fs,'\nMMMF-CG-Weak Testing Error:\tZOE = %.4f\t\tMAE = %.4f\t\tRMSE = %.4f\t\tRRMSE = %.4f',...
    ResultTstMMMFAvg(1,1),ResultTstMMMFAvg(2,1),ResultTstMMMFAvg(3,1),ResultTstMMMFAvg(4,1));




