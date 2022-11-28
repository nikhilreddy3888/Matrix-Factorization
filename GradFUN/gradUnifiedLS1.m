function [obj,grad,lossobj,regobj] = gradUnifiedLS1(v,parameter)

Y               = parameter.Y;              %rating matrix (labels) [n,m], Ytrn
lambda1         = parameter.lambda1;        %Regularization paramete, lambda_1/2 ( (UG-U) + (VG-V))     
lambda3         = parameter.lambda3;        %Regularization paramete, lambda_3/2 (U + V)  
l               = parameter.l;              %Rating Level
d               = parameter.d;              %latent space size
K1              = parameter.K1;             %Number of Clusters in User Space
K2              = parameter.K2;             %Number of Clusters in Item Space
UGidx           = parameter.UGidx;          %User Group IDX
VGidx           = parameter.VGidx;          %Item Group IDX
%%
[n,m]           = size(Y);
U               = reshape(v(1:n*d),n,d);
V               = reshape(v(n*d+1:n*d+m*d),m,d);
UG              = reshape(v(n*d+m*d+1:n*d+m*d+K1*d),K1,d);
VG              = reshape(v(n*d+m*d+K1*d+1:end),K2,d);
clear v;
%%
UGmatMU         = UG(UGidx,:) - U;
VGmatMV         = VG(VGidx,:) - V;
X               = U*V';
Ygt0            = Y > 0;
YMXgt0          = (Y-X).*Ygt0;
%%
regobj          = (lambda3./2).*(sum(U(:).^2) + sum(V(:).^2)); 

lossobj         = 0;
lossobj         = lossobj + (0.5).*sum( YMXgt0(:).^2  );
lossobj         = lossobj + (lambda1./2).* ( sum(UGmatMU(:).^2) + sum(VGmatMV(:).^2));


%%
dU              = lambda3 .* U;       % [n,d]
dV              = lambda3 .* V;       % [m,d]

dU              = dU - YMXgt0*V;            %Gradient From Rating approximation Loss
dV              = dV - YMXgt0'*U;           %Gradient From Rating approximation Loss

dU              = dU - lambda1.* UGmatMU;   %Gradient From Group Loss
dV              = dV - lambda1.* VGmatMV;   %Gradient From Group Loss


% convert UGidx to a K1-by-n matrix containing the k1 indicator vectors as row
ctmp            = sparse(1:n, UGidx, 1)';
dUG             = lambda1.* (ctmp*UGmatMU);

% convert UGidx to a K2-by-m matrix containing the k2 indicator vectors as row
ctmp            = sparse(1:m, VGidx, 1)';
dVG             = lambda1.* (ctmp*VGmatMV);

obj             = regobj + lossobj; % obj is the objective function that we need to minimize
grad            = [dU(:); dV(:); dUG(:); dVG(:)];




