%% interaction kernel  B(u) = \int k k^T hat_V^2/(1+ alpha hat_V)^2 delta (k\dot u) dk
% approximate matrix B for given vector u (neq 0)

% find the support of delta (in the u_parallel direction)
% use gauss-legendre quadrature rule in the u_prep direction





order = 150;
[gauss_nodes, gauss_weights] = gauss_legendre(order); 
k_max = 20;
a = -k_max; b = k_max;
nodes_prep = (a+b)/2 + (b-a)/2*gauss_nodes; 
weights_prep = (b-a)/2 * gauss_weights;
% I = sum(weights_prep.*f(nodes_prep));

delta = 0.02;
n = 3;
nodes_parallel = linspace(-delta, delta, n); 
% quadrature nodes in the rotated coordinate system parallel and perpendicular to u
[rotated_k_parallel, rotated_k_prep] = meshgrid(nodes_parallel, nodes_prep);

norm_k = sqrt(rotated_k_parallel.^2 + rotated_k_prep.^2); % |k| does not change after rotation


% static dielectric fctn on frequency mesh
% this can be precomputed since hat V is radially symmetric
alpha = 0.;
% hat_V = 1./norm_k.^2; %  V^(r) = 1/r^2 Coulomb case
hat_V = 1./(norm_k.^2 + 1); % V^(r) = 1/(1+r^3) add 1 in the denominator to regularize
dielectric = 1 + alpha*hat_V; % equals 1 at the moment


% precompute
tic 
A11 = rotated_k_parallel.^2; A22 = rotated_k_prep.^2; A12 = rotated_k_parallel.*rotated_k_prep; A21 = A12;
rotated_I11 = A11.*hat_V.^2./dielectric.^2;
rotated_I22 = A22.*hat_V.^2./dielectric.^2;
rotated_I12 = A12.*hat_V.^2./dielectric.^2;
toc

theta = -pi/4;
u = [cos(theta); sin(theta)]*3;
norm_u = norm(u);

% rotation matrix: clockwise through an angle theta
rotation = [cos(theta) sin(theta); -sin(theta) cos(theta)];

% quadature nodes in the xOy cartesian coordinate system
k_parallel = rotated_k_parallel * cos(theta) - rotated_k_prep * sin(theta);
k_prep = rotated_k_parallel * sin(theta) + rotated_k_prep * cos(theta);


delta_approx = hat(nodes_parallel, delta);



B11 = k_parallel.^2; B22 = k_prep.^2; B12 = k_parallel.*k_prep; B21 = B12;
I11 = B11.*hat_V.^2./dielectric.^2;
I22 = B22.*hat_V.^2./dielectric.^2;
I12 = B12.*hat_V.^2./dielectric.^2;

I11_new = rotated_I11*cos(theta)^2+rotated_I22*sin(theta)^2-2*rotated_I12*sin(theta)*cos(theta);
I22_new = rotated_I11*sin(theta)^2+rotated_I22*cos(theta)^2+2*rotated_I12*sin(theta)*cos(theta);
I12_new = (rotated_I11-rotated_I22)*sin(theta)*cos(theta)+rotated_I12*(cos(theta)^2-sin(theta)^2);

sum(delta_approx*2*delta/(n-1)) *sum(weights_prep)
sum(sum(delta_approx.*weights_prep*2*delta/(n-1)))
pre = weights_prep.*delta_approx;
kernel11 = sum(sum(I11_new.*pre))/norm_u*2*delta/(n-1);
kernel22 = sum(sum(I22_new.*pre))/norm_u*2*delta/(n-1);
kernel12 = sum(sum(I12_new.*pre))/norm_u*2*delta/(n-1);
Kernel_LB = [kernel11, kernel12; kernel12, kernel22] 

toc

%% kernel of Landau B(u)= L |u|^{gamma+2}(I_d - uu^T/|u|^2), gamma=-3
% with L = w_{d-1}/dw_d \int_{R^d} |k| hat_V(k)^2 dk = w_{d-1} \int_0^inf r^d hat_V(r)^2 dr
Projection = eye(2) - u*u'/norm_u^2; % projection matrix to u
w1 = 2;
w2 = pi;
% discretization in frequency domain
dk = 0.05;
k_max = 20;
k_x = -k_max:dk:k_max;
k_y = -k_max:dk:k_max;
Nk = length(k_x);
[Kx,Ky] = meshgrid(k_x,k_y);
norm_k = sqrt(Kx.^2+Ky.^2);
% ind_k = norm_k>0;
% hat_V = 1./norm_k.^2; 
% L1 = w1/2/w2*sum(sum(norm_k(ind_k).*hat_V(ind_k).^2))*dk^2;
hat_V = 1./(norm_k.^2 + 1); 
L1 = w1/2/w2*sum(sum(norm_k.*hat_V.^2))*dk^2;

% For 2d V^(r) = 1/(1+r^2), \int_0^inf r^2/(1+r^2)^2 dr =  pi/4 = 0.7854
% % r_max at least 60
% For 2d V^(r) = 1/(1+r^3),  \int_0^inf r^2/(1+r^3)^2 dr =  1/3
% % V^(r) decay faster, r_max = 20 is good enough
r_max = k_max;
dr = dk;
rr = dr:dr:r_max;
% hat_Vr = 1./rr.^2; % V^(r) = 1/r^2
hat_Vr = 1./(rr.^2 + 1); % V^(r) = 1/(1+r^3)
intergrand = rr.^2.*hat_Vr.^2;
weight = dr*[0.5, ones(1,length(rr)-2), 0.5];
L2 = w1*sum(intergrand.*weight); % L2 more accurate than L1
Kernel_Landau = L2/norm_u*Projection
err_matrix = Kernel_LB - Kernel_Landau;
err = max(max(abs(err_matrix)))
