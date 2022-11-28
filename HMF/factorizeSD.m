function [y] = factorizeSD(Ytrn,~,n,m,level)
%
    gradfun = @m3fshc_binary; %gradient function
    p = 100; %number of latent factor
    %alpha = 0.01; % constant multiplied with gradient while updation
    %
    if level == 1
        lambda = sqrt(sqrt(10)).^[ 2 ];
    end
    if level == 2
        lambda = sqrt(sqrt(10)).^[ 2.7 ];
        %lambda = 4.6;
    end
    if level == 3
        lambda = sqrt(sqrt(10)).^[ 3.5 ];
        %lambda = 7.7;
    end
    if level == 4
        lambda = sqrt(sqrt(10)).^[ 2.5 ];
    end
    %}
    %lambda = sqrt(sqrt(10)).^[ 2.6 ]; %regularization value;   
    %lambda = 4.1;
    maxiter= 100;
    v = randn(n*p + m*p ,1);

    for i=1:length(lambda)
        [v] = steepestDescent(v,Ytrn,gradfun,p,maxiter,lambda(i),alpha);
        U = reshape(v(1:n*p),n,p);
        V = reshape(v(n*p+1:n*p+m*p),m,p);
        X = U * V';
        y = mapRating(X,n,m);
    end
end