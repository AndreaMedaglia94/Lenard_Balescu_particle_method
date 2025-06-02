function y = mollifier2d(vx,vy,eps)
% psi(v)=exp(-|v|^2/2/eps)/(2*pi*eps)^{d/2} with d=2

y = exp(-(vx.^2+vy.^2)/2/eps)/(2*pi*eps);

end