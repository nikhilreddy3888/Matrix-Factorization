function [v,lp_idx,J] = steepestDescentRLSGD(v,Ytrn,gradfun,k,maxiter,lambda,L)
%U is user feature matrix
%V is movie feature matrix
%Rtrain is training set
%gradfun is function handle for gradient calculation
%p is number of latent feature
%maxiter is maximum iteration upto which the gradient function is updated
%lambda is regularization parameter
%alpha is step size

J = zeros(1,maxiter);
lp_idx = 1;
[n,m] = size(Ytrn);

while lp_idx <= maxiter 

    %tic
    [obj_new,grad] = gradfun(v,Ytrn,lambda,k);
    fprintf('\nJ = %.4f',obj_new);
    J(lp_idx) = obj_new;
    [ alpha, ~, flag ] = lineSearchRLSGD(v, grad, obj_new, Ytrn, lambda, k);
    if flag == 1
        break;
    else
        v = v - alpha .* grad;
        lp_idx = lp_idx + 1;
    end
    
    %toc
end


clear  obj_old alpha allowNonDecrease U_old V_old
end