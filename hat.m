function y = hat(x, eps)
y = 1/eps*(1-abs(x)/eps).*(abs(x)<eps);
end