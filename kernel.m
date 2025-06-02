function [kernel11,kernel12,kernel22] = kernel(ux, uy, rotated_I11, rotated_I12, rotated_I22, pre)
% for given u, compute B(u) = int Phi(k) delta (k\cdot u) dk 
%                           = 1/|u| int R^T Phi(p) R delta(p_//) dp 
% Phi(p) = [rotated_I11, rotated_I12; rotated_I12, rotated_I22] was precomputed at quadrature nodes

% rotation matrix R = (cos t, sint; -sin t , cos t)
norm_u = sqrt(ux^2+uy^2);
cos_theta = ux/norm_u;
sin_theta = uy/norm_u;
                            
% Phi(k) = k \otimes k (V^(k)/(1+alpha V^(k)))^2 = R^T Phi(p) R
I11 = rotated_I11*cos_theta^2+rotated_I22*sin_theta^2-2*rotated_I12*sin_theta*cos_theta;
I22 = rotated_I11*sin_theta^2+rotated_I22*cos_theta^2+2*rotated_I12*sin_theta*cos_theta;
I12 = (rotated_I11-rotated_I22)*sin_theta*cos_theta+rotated_I12*(cos_theta^2-sin_theta^2);


kernel11 = sum(sum(I11.*pre))/norm_u;
kernel22 = sum(sum(I22.*pre))/norm_u;
kernel12 = sum(sum(I12.*pre))/norm_u;

end