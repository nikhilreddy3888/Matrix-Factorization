
function [R_final,hmf_fnl_trn_err,hmf_fnl_tst_err] = hmf_seven(Rtrain,Rtest,n,m,flagSdCg)

hmf_fnl_trn_err = zeros(1,3);  
hmf_fnl_tst_err = zeros(1,3);  

id1 = (Rtrain==1); %find(R==1);
id2 = (Rtrain==2); %find(R==2);
id3 = (Rtrain==3); %find(R==3);
id4 = (Rtrain==4); %find(R==4);
id5 = (Rtrain==5); %find(R==5);

%{
id1_test = (Rtest==1); 
id2_test = (Rtest==2); 
id3_test = (Rtest==3); 
id4_test = (Rtest==4); 
id5_test = (Rtest==5); 
%}

RCAP = zeros(n,m);
RCAP(id1|id2) = -1;
RCAP(id4|id5) = 1;

%RCAP_test = zeros(n,m);
%RCAP_test(id1_test|id2_test) = -1;
%RCAP_test(id4_test|id5_test) = 1;

%CODE FOR DIVIDING DATA INTO TRAINING SET  AND TESTING SET
level = 1;
[p, ~,~] = setParameter();
v = randn(n*p + m*p ,1);

[y]=factorize(RCAP,n,m,level,flagSdCg,v);
%UniqueU_MMMF(RCAP,U,V,theta)

Rcap_ind = (y == -1); %ucap . vcap<=theta

R1 = zeros(n,m);
R1(id1) = -1;
R1(id3) = 1;

%R1_test = zeros(n,m);
%R1_test(id1_test) = -1;
%R1_test(id3_test) = 1;

R1 = Rcap_ind .* R1; %ucap . vcap <= theta and if Rij==1 then 1 and if Rij=3 then 2 else 0
%R1_test = R1_test .* Rcap_ind;
R10 = (R1 == 0);
R1id3 = (R1 == 1);

R2 = zeros(n,m);
R2(id3) = -1;
R2(id5) = 1;

%R2_test = zeros(n,m);
%R2_test(id3_test) = -1;
%R2_test(id5_test) = 1;

R2 = ~Rcap_ind .* R2;%ucap . vcap >= theta and if Rij==3 then 1 and if Rij=5 then 2 else 0
%R2_test = R2_test .* ~Rcap_ind;

R20 = (R2 == 0);
R2id3 = (R2 == -1);

R1( R10 & R2id3 ) = 1;
level = 2;
[y1] = factorize(R1,n,m,level,flagSdCg,v);

level = 3;
R2( R20 & R1id3 ) = -1;
[y2] = factorize(R2,n,m,level,flagSdCg,v);


y1 = y1 .* Rcap_ind;  %R'ij ~=0
y2 = y2 .* ~Rcap_ind; %~R'ij ~=0

R1_ind1 = (y1 == -1); %u' . v'<=theta
R1_ind2 = (y1 == 1); %u' . v'>theta

R2_ind1 = (y2 == -1); %u2 . v2<=theta
R2_ind2 = (y2 == 1); %u2 . v2>theta

%id3Final = (R1_ind2 & R2_ind1);

R3 = zeros(n,m);
R3(id1) = -1;
R3(id2) = 1;
R3 = R1_ind1 .* R3; %u' . v' <= theta and R'ij ~=0 and if Rij==1 then 1 and if Rij=2 then 2 else 0

R30 = (R3 == 0);
R3id2 = (R3 == 1);
%R3_test = zeros(n,m);
%R3_test(id1_test) = -1;
%R3_test(id2_test) = 1;
%R3_test = R3_test .* R1_ind1;

R4 = zeros(n,m);
R4(id2) = -1;
R4(id3) = 1;
R4 = R1_ind2 .* R4;%u' . v' > theta and R'ij ~=0 and if Rij==2 then 1 and if Rij=3 then 2 else 0

R40 = (R4 == 0);
R4id2 = (R4 == -1);

R3( R30 & R4id2 ) = 1;
R4( R40 & R3id2 ) = -1;
%R4_test = zeros(n,m);
%R4_test(id2_test) = -1;
%R4_test(id3_test) = 1;
%R4_test = R4_test .* R1_ind2;

R40 = (R4 == 0);
R4id3 = (R4 == 1);

R5 = zeros(n,m);
R5(id3) = -1;
R5(id4) = 1;
R5 = R2_ind1 .* R5; %u2 . v2 <= theta and R2ij ~=0 and if Rij==3 then 1 and if Rij=4 then 2 else 0

R50 = (R5 == 0);
R5id3 = (R5 == -1);

R4( R40 & R5id3 ) = 1;
R5( R50 & R4id3 ) = -1;

R50 = (R5 == 0);
R5id4 = (R5 == 1);

%R5_test = zeros(n,m);
%R5_test(id3_test) = -1;
%R5_test(id4_test) = 1;
%R5_test = R5_test .* R2_ind1;

R6 = zeros(n,m);
R6(id4) = -1;
R6(id5) = 1;
R6 = R2_ind2 .* R6; %u2 . v2 > theta and R2ij ~=0 and if Rij==4 then 1 and if Rij=5 then 2 else 0

R60 = (R6 == 0);
R6id4 = (R6 == -1);

R5( R50 & R6id4 ) = 1;

R6( R60 & R5id4 ) = -1;
%R6_test = zeros(n,m);
%R6_test(id4_test) = -1;
%R6_test(id5_test) = 1;
%R6_test = R6_test .* R2_ind2;

level = 4;
[y3]=factorize(R3,n,m,level,flagSdCg,v);

level = 5;
[y4]=factorize(R4,n,m,level,flagSdCg,v);

level = 6;
[y5]=factorize(R5,n,m,level,flagSdCg,v);

level = 7;
[y6]=factorize(R6,n,m,level,flagSdCg,v);


y3 = y3 .* R1_ind1;
y4 = y4 .* R1_ind2;
y5 = y5 .* R2_ind1;
y6 = y6 .* R2_ind2;

R_final = zeros(n,m);

R_final(y3 == -1) = 1;
R_final(y3 == 1) = 2;
R_final(y4 == -1) = 2;
R_final(y4 == 1) = 3;
R_final(y5 == -1) = 3;
R_final(y5 == 1) = 4;
R_final(y6 ==-1) = 4;
R_final(y6 == 1) = 5;

%R_final(id3Final==1) = 3;
%err
%fprintf(1,' ZOE: %.6f %.6f MAE: %.6f  %.6f\n',zoe(R_final,Rtrain),zoe(R_final,Rtest),mae(R_final,Rtrain),mae(R_final,Rtest));
hmf_fnl_trn_err(1) = zoe(R_final,Rtrain);
hmf_fnl_trn_err(2) = mae(R_final,Rtrain);
hmf_fnl_trn_err(3) = RMSE(R_final,Rtrain);

hmf_fnl_tst_err(1) = zoe(R_final,Rtest);
hmf_fnl_tst_err(2) = mae(R_final,Rtest); 
hmf_fnl_tst_err(3) = RMSE(R_final,Rtest);
end