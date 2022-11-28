function [ret] = hloss(z)
  zle1 = z < 1;
  ret =  1 .* zle1 - z .* zle1  ;
end