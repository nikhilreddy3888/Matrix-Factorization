function [RFinal,ULevelWise,VLevelWise,hmf_Lvl_trn_err,hmf_Lvl_tst_err] ...
    = hmfTstWeak(Rtrn, Rtst, par)

[n,m] = size(Rtrn);
p = par.p;
ratingLevel = par.ratingLevel;
increment = par.increment;
minRating = par.minRating;

hmf_Lvl_trn_err = zeros(1,(ratingLevel/increment)-1 );  %variable used to store level-wise training error               
hmf_Lvl_tst_err = zeros(1,(ratingLevel/increment)-1 );  %variable used to store level-wise testing error

hmf_fnl_trn_err = zeros(1,5);  %variable used to store final training error
hmf_fnl_tst_err = zeros(1,5);  %variable used to store final testing error

RFinal = zeros(n,m);

ULevelWise = zeros(n,p,ratingLevel-1);
VLevelWise = zeros(m,p,ratingLevel-1);
i = minRating;
while i < ratingLevel
%for i= priorityOrder(1:ratingLevel-1)
        level = i;
        %code for converting into binary rating matrix
        RcapTrn = zeros(n, m);
        RcapTst = zeros(n, m);
        
        RcapTrn( (Rtrn <= i) & ( Rtrn ~= 0 ) ) = -1;
        RcapTst( (Rtst <= i) & ( Rtst ~= 0 ) ) = -1;
        
        RcapTrn( (Rtrn > i) & ( Rtrn ~= 0 ) ) = 1;
        RcapTst( (Rtst > i) & ( Rtst ~= 0 ) ) = 1;
        
        RcapTrn = sparse(RcapTrn);
        RcapTst = sparse(RcapTst);
        
        [y, U, V]=factorizeWeak(RcapTrn, level/increment, par);
        ULevelWise(:,:,i) = U;
        VLevelWise(:,:,i) = V;
        %y
        %full(RcapTrn)
        %clear v;
        hmf_Lvl_trn_err(i/increment) = zoe(y,RcapTrn);
        hmf_Lvl_tst_err(i/increment) = zoe(y,RcapTst);
        RFinal( y == -1 & RFinal == 0 ) = i;
        i = i + increment;
        clear RcapTst RcapTrn;
end

RFinal( RFinal == 0 ) = ratingLevel;
%RFinal( RFinal == 0 ) = priorityOrder(ratingLevel);
clear y RcapTrn RcapTst clear v;

hmf_fnl_trn_err(1) = zoe(RFinal,Rtrn);
hmf_fnl_trn_err(2) = mae(RFinal,Rtrn);
hmf_fnl_trn_err(3) = RMSE(RFinal,Rtrn);
hmf_fnl_trn_err(4) = averageMAE(RFinal,Rtrn);
hmf_fnl_trn_err(5) = averageRMSE(RFinal,Rtrn);

hmf_fnl_tst_err(1) = zoe(RFinal,Rtst);
hmf_fnl_tst_err(2) = mae(RFinal,Rtst); 
hmf_fnl_tst_err(3) = RMSE(RFinal,Rtst);
hmf_fnl_tst_err(4) = averageMAE(RFinal,Rtst); 
hmf_fnl_tst_err(5) = averageRMSE(RFinal,Rtst);
end

        
    
    
    