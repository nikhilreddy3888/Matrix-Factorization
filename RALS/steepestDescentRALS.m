function [U,V,lp_idx] = steepestDescentRALS(U,V,Ytrn,gradfunU,gradfunV,maxiter,lambda)

%U is user feature matrix
%V is movie feature matrix
%Rtrain is training set
%gradfun is function handle for gradient calculation
%p is number of latent feature
%maxiter is maximum iteration upto which the gradient function is updated
%lambda is regularization parameter
%alpha is step size


lp_idx = 1;
maxepoch = 50;

obj_old = inf;

while lp_idx <= maxiter
    %fprintf('\nIteration: %d',lp_idx);

    for lp_idxU=1: maxepoch
        [obj_new,dU] = gradfunU(U,V,Ytrn,lambda);
        fprintf('\nJ = %.4f',obj_new);
        [ alpha, flag ] = lineSearchRALSU(U,V, dU, obj_new, Ytrn, lambda);
        if flag == 1
            break;
        else
            U = U - alpha .* dU;
        end
  
    end
    %pause(3);
    for lp_idxV=1: maxepoch
        
        [obj_new,dV] = gradfunV(U,V,Ytrn,lambda);
        fprintf('\nJ = %.4f',obj_new);
        [ alpha, flag ] = lineSearchRALSV(U,V, dV, obj_new, Ytrn, lambda);
        if flag == 1
            break;
        else
            V = V - alpha .* dV;
        end
    end
    
    lp_idx = lp_idx + 1;
    
    %toc
end


clear  obj_old alpha allowNonDecrease U_old V_old
end

