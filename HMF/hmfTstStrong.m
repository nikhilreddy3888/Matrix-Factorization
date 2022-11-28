function [RFinal,hmfLvlTrnErr,hmfLvlTstErr,hmfFnlTrnErr,hmfFnlTstErr ] ...
    = hmfTstStrong(weakTrn, weakTst, strongTrn, strongTst, flagSdCg,lambda)

[~, ~, ~, ratingLevel, increment, minRating] = setParameter();
[n, m] = size(weakTrn);
[n1, m1] = size(strongTrn);

hmfLvlTrnErr = zeros(1,(ratingLevel/increment)-1 );  %variable used to store level-wise training error               
hmfLvlTstErr = zeros(1,(ratingLevel/increment)-1 );  %variable used to store level-wise testing error
hmfFnlTrnErr = zeros(1,5);  %variable used to store final training error
hmfFnlTstErr = zeros(1,5);  %variable used to store final testing error

weakAll = weakTrn + weakTst;
RFinal = zeros(n1,m1);
clear weakTrn weakTst;

%For first level U,V and U',V' is random 
%[p, ~, ~, ~, ~, ~] = setParameter();
%v = randn(n*p + m*p ,1);
%w = randn(n1*p, 1); 

i = minRating;
while i < ratingLevel
        level = i;
        weakAllCap = zeros(n, m);
        strongCapTrn = zeros(n1,m1);
        strongCapTst = zeros(n1,m1);
        
        %code for converting into binary rating matrix
        
        weakAllCap( (weakAll <= i) & ( weakAll ~= 0 ) ) = -1;
        strongCapTrn( (strongTrn <= i) & ( strongTrn ~= 0 ) ) = -1;
        strongCapTst( (strongTst <= i) & ( strongTst ~= 0 ) ) = -1;
                
        weakAllCap( (weakAll > i) & ( weakAll ~= 0 ) ) = 1;
        strongCapTrn( (strongTrn > i) & ( strongTrn ~= 0 ) ) = 1;
        strongCapTst( (strongTst > i) & ( strongTst ~= 0 ) ) = 1;
        
        weakAllCap = sparse(weakAllCap);
        strongCapTrn = sparse(strongCapTrn);
        strongCapTst = sparse(strongCapTst);

        [y]=factorizeStrong(weakAllCap, strongCapTrn, level/increment, flagSdCg,lambda);

        hmfLvlTrnErr(i/increment) = zoe(y,strongCapTrn);
        hmfLvlTstErr(i/increment) = zoe(y,strongCapTst);
        RFinal( y == -1 & RFinal == 0 ) = i;
        
        clear y strongCapTrn strongCapTst weakAllCap;
        i = i + increment;
        
end

RFinal( RFinal == 0 ) = ratingLevel;

%clear v w;
hmfFnlTrnErr(1) = zoe(RFinal,strongTrn);
hmfFnlTrnErr(2) = mae(RFinal,strongTrn);
hmfFnlTrnErr(3) = RMSE(RFinal,strongTrn);
hmfFnlTrnErr(4) = averageMAE(RFinal,strongTrn);
hmfFnlTrnErr(5) = averageRMSE(RFinal,strongTrn);

hmfFnlTstErr(1) = zoe(RFinal,strongTst);
hmfFnlTstErr(2) = mae(RFinal,strongTst); 
hmfFnlTstErr(3) = RMSE(RFinal,strongTst);
hmfFnlTstErr(4) = averageMAE(RFinal,strongTst); 
hmfFnlTstErr(5) = averageRMSE(RFinal,strongTst);

end

        
    
    
    