function [obj,grad,lossobj,regobj] = gradGMMMF(v,parameter)
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


K1              = parameter.K1;             %Number of Clusters in User Space
UGidx           = parameter.UGidx;          %User Group IDX
d               = parameter.d;              %latent space size
l               = parameter.l;

[n,m]           = size(Y);


UG              = reshape(v(1:K1*d),K1,d);
V               = reshape(v(K1*d+1:K1*d+m*d),m,d);
theta           = reshape(v(K1*d+m*d+1:K1*d+m*d + K1*(l-1)),K1,l-1);

thetaM          = theta(UGidx,:);

clear v;

UGM             = UG(UGidx,:);
X               = UGM*V';
Ygt0            = Y>0;
BX              = X.*Ygt0;




clear X;

dUG             = lambda1.*UG; % [n,p]
dV              = lambda1.*V; % [m,p]
dtheta          = zeros(K1,l-1); % [n,l-1]

regobj = lambda1.*(sum(UG(:).^2)+sum(V(:).^2))./2; % [scalar]
lossobj = 0;
for k=1:l-1
    S = Ygt0-2.*(Y>k); %S is T in the paper
    % Next line is the memory bottleneck
    BZ = (thetaM(:,k)*ones(1,m)).*S - BX.*S; % [n,m] (sparse)
    lossobj = lossobj + sum(sum(h(BZ)));
    tmp = hprime(BZ).*S; % [n,m]
    clear BZ S;
    
    ctmp            = sparse(1:n, UGidx, 1)';
    
    dUG = dUG - ctmp*(tmp*V); % [n,p]
    
    dV = dV - tmp'*UGM; % [m,p]
    dtheta(:,k) = ctmp*(tmp*ones(m,1));
    clear tmp;
end


obj = regobj + lossobj; % obj is the objective function that we need to minimize
grad = [dUG(:); dV(:); dtheta(:)];
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
