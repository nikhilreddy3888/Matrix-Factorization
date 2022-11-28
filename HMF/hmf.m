function [RFinal,hmf_Lvl_trn_err,hmf_Lvl_tst_err,hmf_fnl_trn_err,hmf_fnl_tst_err ] = hmf(Rtrn, Rtst, n, m, flagSdCg)

hmf_Lvl_trn_err = zeros(1,4);  %variable used to store level-wise training error               
hmf_Lvl_tst_err = zeros(1,4);  %variable used to store level-wise training error

hmf_fnl_trn_err = zeros(1,5);  %variable used to store final training error
hmf_fnl_tst_err = zeros(1,5);  %variable used to store final testing error
 
%keeping track of non-zero entry in training data
id1 = (Rtrn==1); 
id2 = (Rtrn==2); 
id3 = (Rtrn==3); 
id4 = (Rtrn==4); 
id5 = (Rtrn==5);
%keeping track of non-zero entry in testing data
id1_test = (Rtst==1); 
id2_test = (Rtst==2); 
id3_test = (Rtst==3); 
id4_test = (Rtst==4); 
id5_test = (Rtst==5); 

%first level training data
level = 1;
R1 = zeros(n,m);
R1(id1) = -1;
R1(id2|id3|id4|id5) = 1;
%first level testing data
R1_test = zeros(n,m);
R1_test(id1_test) = -1;
R1_test(id2_test|id3_test|id4_test|id5_test) =1;
%first level factorization
[p, ~, ~, ~, ~, ~] = setParameter();
v = randn(n*p + m*p ,1);

[y1,v]=factorize(R1,n,m,level,flagSdCg,v);
%calculation of first level training error
hmf_Lvl_trn_err(1) = zoe(y1,R1);
%calculation of first level testing error
hmf_Lvl_tst_err(1) = zoe(y1, R1_test);
%index of rating 1 in final matrix
R1_ind = (y1 == -1); 

clear R1 R1_test;

%second level training data
level = 2;
R2 = zeros(n,m);
R2(id1|id2) = -1;
R2(id3|id4|id5) = 1;
%second level testing data
R2_test = zeros(n,m);
R2_test(id1_test|id2_test) = -1;
R2_test(id3_test|id4_test|id5_test) = 1;
%second level factorization
[y2,v] = factorize(R2,n,m,level,flagSdCg,v);
%calculation of second level training error
hmf_Lvl_trn_err(2)=zoe(y2,R2);
%calculation of first level testing error
hmf_Lvl_tst_err(2) = zoe(y2, R2_test);

%index of rating 2 in final matrix
y2 = y2 .* ~R1_ind;  
R2_ind = (y2 == -1); 

clear R2 R2_test;


%third level training data
level = 3;
R3 = zeros(n,m);
R3(id1|id2|id3) = -1;
R3(id4|id5) = 1;
%third level training data
R3_test = zeros(n,m);
R3_test(id1_test|id2_test|id3_test) = -1;
R3_test(id4_test|id5_test) = 1;
%third level factorization
[y3,v] = factorize(R3,n,m,level,flagSdCg,v);
%calculation of third level training error
hmf_Lvl_trn_err(3)=zoe(y3,R3);
%calculation of third level testing error
hmf_Lvl_tst_err(3) = zoe(y3, R3_test);

%index of rating 3 in final matrix
y3 = y3 .* ~(R1_ind|R2_ind); 
R3_ind = (y3 == -1);

clear R3 R3_test;

%fourth level training data
level = 4;
R4 = zeros(n,m);
R4(id1|id2|id3|id4) = -1;
R4(id5) = 1;
%fourth level testing data
R4_test = zeros(n,m);
R4_test(id1_test|id2_test|id3_test|id4_test) = -1;
R4_test(id5_test) = 1;
%fourth level factorization
[y4,v] = factorize(R4,n,m,level,flagSdCg,v);
%calculation of fourth level testing error
hmf_Lvl_trn_err(4) = zoe(y4, R4);

%calculation of fourth level testing error
hmf_Lvl_tst_err(4) = zoe(y4, R4_test);

%index of rating 4 in final matrix
y4 = y4 .* ~(R1_ind|R2_ind|R3_ind);
R4_ind = (y4 == -1);

%index of rating 3 in final matrix
R5_ind = (y4 == 1);

clear R4 R4_test v;

RFinal = zeros(n,m);
RFinal(R1_ind) = 1;
RFinal(R2_ind) = 2;
RFinal(R3_ind) = 3;
RFinal(R4_ind) = 4; 
RFinal(R5_ind) = 5;

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

%R = Rtrain + Rtest;
%hmf_fnl_tst_err(4) = rre( R_final, R);
%}
%err, test_err
%fprintf(1,' ZOE: %.6f %.6f MAE: %.6f  %.6f\n',zoe(R_final,Rtrain),zoe(R_final,Rtest),mae(R_final,Rtrain),mae(R_final,Rtest));
end


%change log
%converted -1 to 1 and +1 to 2
