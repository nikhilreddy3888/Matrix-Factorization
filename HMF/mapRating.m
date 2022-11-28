function [y] = mapRating( X,n,m)

y = zeros(n,m);
y( X < 0 ) = -1;
y( X > 0 ) = 1;
end