function [obj,grad,lossobj,regobj] = gradUnifiedMMMF1(v,parameter)
fn = mfilename;
if nargin < 2
    error('insufficient parameters')
end
% Parameters that can be set via varargin
%verbose = 1;
% Process varargin
%paramgt;
Y = parameter.Y;
lambda1         = parameter.lambda1;        %Regularization paramete, lambda_3/2 (U + V)
lambda3         = parameter.lambda3;        %Regularization paramete, lambda_1/2 ( (UG-U) + (VG-V))


K1              = parameter.K1;             %Number of Clusters in User Space
K2              = parameter.K2;             %Number of Clusters in Item Space
UGidx           = parameter.UGidx;          %User Group IDX
VGidx           = parameter.VGidx;          %Item Group IDX
d               = parameter.d;              %latent space size
l               = parameter.l;

[n,m]           = size(Y);


U               = reshape(v(1:n*d),n,d);
V               = reshape(v(n*d+1:n*d+m*d),m,d);
theta           = reshape(v(n*d+m*d+1:n*d+m*d+n*(l-1)),n,l-1);

UG              = sparse(reshape(v(n*d+m*d+n*(l-1)+1:n*d+m*d+n*(l-1)+K1*d),K1,d));
VG              = sparse(reshape(v(n*d+m*d+n*(l-1)+K1*d+1:n*d+m*d+n*(l-1)+K1*d + K2*d),K2,d));
thetaG          = sparse(reshape(v(n*d+m*d+n*(l-1)+K1*d + K2*d+1:end),K1,l-1)); % different from gradGroupRSMMMF



clear v;

X               = U*V';
Ygt0            = Y>0;
BX              = X.*Ygt0;
UGmatMU         = UG(UGidx,:) - U;
VGmatMV         = VG(VGidx,:) - V;
thetaGMtheta    = thetaG(UGidx,:) - theta;

clear X;

dU = lambda3.*U; % [n,p]
dV = lambda3.*V; % [m,p]
dtheta = zeros(n,l-1); % [n,l-1]
regobj = lambda3.*(sum(U(:).^2)+sum(V(:).^2))./2; % [scalar]
lossobj = 0;
lossobj         = lossobj + (lambda1./2).* ( sum(UGmatMU(:).^2) + sum(VGmatMV(:).^2) + sum(thetaGMtheta(:).^2));
for k=1:l-1
    S = Ygt0-2.*(Y>k); %S is T in the paper
    % Next line is the memory bottleneck
    BZ = (theta(:,k)*ones(1,m)).*S - BX.*S; % [n,m] (sparse)
    lossobj = lossobj + sum(sum(h(BZ)));
    tmp = hprime(BZ).*S; % [n,m]
    clear BZ S;
    dU = dU - tmp*V; % [n,p]
    dV = dV - tmp'*U; % [m,p]
    dtheta(:,k) = tmp*ones(m,1);
    clear tmp;
end

dU              = dU - lambda1.* UGmatMU;   %Gradient From Group Loss
dV              = dV - lambda1.* VGmatMV;   %Gradient From Group Loss
dtheta          = dtheta - lambda1.* thetaGMtheta;

% convert UGidx to a K1-by-n matrix containing the k1 indicator vectors as row
ctmp            = sparse(1:n, UGidx, 1)';
dUG             = lambda1.* (ctmp*UGmatMU);

dthetaG         = lambda1.* (ctmp*thetaGMtheta);

% convert UGidx to a K2-by-m matrix containing the k2 indicator vectors as row
ctmp            = sparse(1:m, VGidx, 1)';
dVG             = lambda1.* (ctmp*VGmatMV);


obj = regobj + lossobj; % obj is the objective function that we need to minimize
grad = [dU(:); dV(:); dtheta(:); dUG(:); dVG(:);dthetaG(:)];
%   if verbose
%     fprintf(1,'%s: lambda=%.2e obj=%.2e grad''*grad=%.2e time=%.1f\n',fn,lambda,obj,grad'*grad,etime(clock,t0));
%   end

% ret = (z>0).*(z<1).*((1-z).^2)/2.0 + (z<=0).*(0.5-z);
function [ret] = h(z)
zin01 = (z>0)-(z>=1);
zle0 = z<0;
ret = zin01./2 - zin01.*z + zin01.*z.^2./2 + zle0./2 - zle0.*z;

% ret = (z>0).*(z<1).*(z-1) - (z<=0);
function [ret] = hprime(z)
zin01 = (z>0)-(z>=1);
zle0 = z<0;
ret = zin01.*z - zin01 - zle0;

% ChangeLog
% 12/8/06 - remove reallyzero (print warning if it would be necessary)
% 3/23/05 - made calcultions take better advantage of sparseness
% 3/18/05 - fixed bug in objective (wasn't squaring fro norms)
% 3/1/05 - added objective calculation
% 2/23/05 - fixed bug in hprime()
