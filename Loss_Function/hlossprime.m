function [ret] = hlossprime(z)
  zle1 = z < 1;
  ret =  - zle1;
end