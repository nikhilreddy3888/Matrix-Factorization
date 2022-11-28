function [ret] = hsmooth(z)
  zin01 = (z>0)-(z>=1);
  zle0 = z<=0;
  ret = ( ( (zin01./2 - zin01.*z ) + zin01.*z.^2./2 ) + zle0./2 ) - zle0.*z;
  clear zin01 zle0
end