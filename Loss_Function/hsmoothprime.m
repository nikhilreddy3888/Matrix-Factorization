function [ret] = hsmoothprime(z)
  zin01 = (z>0)-(z>1);
  zle0 = z<=0;
  ret = zin01.*z - zin01 - zle0;
  clear zin01 zle0
end