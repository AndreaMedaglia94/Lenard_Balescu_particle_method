% 2D homogeneous Lenard_Balescu with static dielectric function
% particle method
% random batch
% energy conservation

set(0,'defaultaxesfontsize',20,'defaultaxeslinewidth',1.5,...
    'defaultlinelinewidth',3,'defaultpatchlinewidth',.7)

%% parameters
d = 2; % dimension

Nv = 40; % number of velocity pts per dim 
v_max = 4; % range of velocity [-v_max, v_max]^d

T = 10; % terminal time
dt = 0.05; % time step
power = 1.98; 


h = 2*v_max/Nv;
eps = 0.64*h^power; % regularization parameter
N = Nv^d; % particle number

% random batch setup
q = 5; % number of batch per dimension
p = Nv/q; % batch size per dimension
RBM_num = (N-1)/(p^d-1);

% parameters in frequency domain
% delta function approximation delta(k\dot u)
order = 150; % order of quadrature rule in k perpendicular to u
k_max = 20; % cut off of k
delta = 0.02; % width of shapefunction 
m = 3; % number of pts in k parallel to u

% static dielectric function: dielectric = 1 + alpha*V^; 
% when alpha = 0, it reduces to the classical Landau
alpha = 0.; 

%% preparation

% time discretization
tt = 0:dt:T;
Nt = length(tt);

% centers of velocity mesh at which to reconstuct f
vc_x = -v_max+h/2:h:v_max;
vc_y = -v_max+h/2:h:v_max;
[Vcx,Vcy] = meshgrid(vc_x,vc_y);
norm_v_square = Vcx.^2+Vcy.^2;

% initialization of particle method
% uniformly distributed in the velocity mesh with weights assigned according to some given initial distribution
Lp = v_max;
hp = 2*Lp/Nv;
v_x = -Lp+hp/2:hp:Lp;
v_y = -Lp+hp/2:hp:Lp;
[Vx0,Vy0] = meshgrid(v_x,v_y);
norm_p_square = Vx0.^2+Vy0.^2;
% % taken from example 1 in particle Landau paper
K0 = 1/2;
f0 = 1/2/pi/K0*exp(-norm_p_square/2/K0).*((2*K0-1)/K0+(1-K0)/2/K0^2*norm_p_square);
% % taken from example 3 in particle Landau paper
% u1 = [-2 1];
% u2 = [0 -1];
% f0 = (exp(-((Vx0-u1(1)).^2+(Vy0-u1(2)).^2)/2)+exp(-((Vx0-u2(1)).^2+(Vy0-u2(2)).^2)/2))/4/pi;
weight = f0*hp^d;
target_energy = sum(sum(weight.*norm_p_square));
target_momentum_x = sum(sum(weight.*Vx0)); target_momentum_y = sum(sum(weight.*Vy0));
target_variance = sum(sum(weight.*((Vx0-target_momentum_x).^2+(Vy0-target_momentum_y).^2)));

% particle solution at the square centers
Vx = Vx0; Vy = Vy0;
f = zeros(Nv,Nv);
for i = 1:Nv
    for j = 1:Nv
        f(i,j) = sum(sum(weight.*mollifier2d(Vcx(i,j)-Vx,Vcy(i,j)-Vy,eps)));
    end
end

% initialization of quantities of interest
energy = zeros(1,Nt);
entropy = zeros(1,Nt);
L2error = zeros(1,Nt);
group_velocity_x = zeros(1,Nt); group_velocity_y = zeros(1,Nt);
maxparticle_x = zeros(1,Nt); maxparticle_y = zeros(1,Nt);
out_of_region = zeros(1,Nt);
f_slice_x = zeros(Nt,Nv); f_slice_y = zeros(Nt,Nv); 

% conserved quantities
energy(1) = target_energy;
group_velocity_x(1) = target_momentum_x; group_velocity_y(1) = target_momentum_y;
entropy(1) = sum(sum(f.*log(f)))*h^d;
maxparticle_x(1) = max(max(abs(Vx))); maxparticle_y(1) = max(max(abs(Vy)));
% record the cross-sections f(0,v_y) and f(v_x,0)
f_slice_x(1,:) = f(round(Nv/2),:); 
f_slice_y(1,:) = f(:,round(Nv/2));


% use gauss-legendre quadrature rule in the u_prep direction
% gauss-legendre quadrature nodes and weights for intergal in [-k_max, k_max]
[gauss_nodes, gauss_weights] = gauss_legendre(order);
a = -k_max; b = k_max;
nodes_prep = (a+b)/2 + (b-a)/2*gauss_nodes;
weights_prep = (b-a)/2 * gauss_weights;

% use hat function to approximate delta fctn in u_parallel
d_delta = 2*delta/(m-1); % weights_parallel
nodes_parallel = linspace(-delta, delta, m);
delta_approx = hat(nodes_parallel, delta);
pre = weights_prep.*delta_approx*d_delta;

% quadrature nodes in the rotated coordinate system (parallel and perpendicular to u)
[rotated_k_parallel, rotated_k_prep] = meshgrid(nodes_parallel, nodes_prep);

norm_k = sqrt(rotated_k_parallel.^2 + rotated_k_prep.^2); % |k| does not change after rotation

% static dielectric fctn on frequency mesh in the rotated coordinate system
% this can be precomputed since hat V is radially symmetric
% hat_V = 1./norm_k.^2; %  V^(r) = 1/r^2 Coulomb case
hat_V = 1./(norm_k.^3 + 1); % V^(r) = 1/(1+r^3) add 1 in the denominator to regularize
dielectric = 1 + alpha*hat_V; % equals 1 at the moment

% precompute Phi(p) = p \otimes p (V^(p)/(1+alpha V^(p)))^2
tic
A11 = rotated_k_parallel.^2; A22 = rotated_k_prep.^2; A12 = rotated_k_parallel.*rotated_k_prep; A21 = A12;
rotated_I11 = A11.*hat_V.^2./dielectric.^2;
rotated_I22 = A22.*hat_V.^2./dielectric.^2;
rotated_I12 = A12.*hat_V.^2./dielectric.^2;
toc



Vx_temp = zeros(size(Vx)); Vy_temp = zeros(size(Vx));
tic
for iit = 2:Nt
    
    % compute sum h^d \grad psi (vi-vc)log fc
    Fx = zeros(Nv,Nv); Fy = zeros(Nv,Nv);
    for i = 1:Nv
        for j = 1:Nv
            temp = mollifier2d(Vx(i,j)-Vcx,Vy(i,j)-Vcy,eps).*log(f);
            Fx(i,j) = -sum(sum((temp.*(Vx(i,j)-Vcx))))*h^d/eps;
            Fy(i,j) = -sum(sum((temp.*(Vy(i,j)-Vcy))))*h^d/eps;
        end
    end
    
    
    
    % divide into q^2 batches
    index_x=randperm(Nv);
    index_y=randperm(Nv);
    for nbx = 1:q
        for nby = 1:q
            
            ind_x = index_x(p*(nbx-1)+1:p*nbx);
            ind_y = index_y(p*(nby-1)+1:p*nby);
          
            % for i,j in the same batch, u = v_i - v_j
            % B(u) = \int k k^T hat_V^2/(1+hat_V)^2 delta (k\dot u) dk for u = v_i - v_j, forall i,j = 1,...,n
            % box approximation: delta (k \dot u) = 1/2delta/2k_max, |k dot u | <=delta, 0, otherwise.
            kernel11 = zeros(p,p,p,p); kernel12 = zeros(p,p,p,p); kernel22 = zeros(p,p,p,p);
            for i = 1:p
                for j = 1:p
                    
                    for k = 1:i
                        for l = (j+1):p
                            ux = Vx(ind_y(i),ind_x(j))-Vx(ind_y(k),ind_x(l));
                            uy = Vy(ind_y(i),ind_x(j))-Vy(ind_y(k),ind_x(l));
                            [kernel11(i,j,k,l),kernel12(i,j,k,l),kernel22(i,j,k,l)] = kernel(ux, uy, rotated_I11, rotated_I12, rotated_I22, pre);           
                        end
                    end
                    for k = (i+1):p
                        for l = j:p
                            ux = Vx(ind_y(i),ind_x(j))-Vx(ind_y(k),ind_x(l));
                            uy = Vy(ind_y(i),ind_x(j))-Vy(ind_y(k),ind_x(l));
                            [kernel11(i,j,k,l),kernel12(i,j,k,l),kernel22(i,j,k,l)] = kernel(ux, uy, rotated_I11, rotated_I12, rotated_I22, pre);           
                        end
                    end
                    % fill the other half by symmetry
                    kernel11(:,:,i,j) = kernel11(i,j,:,:);
                    kernel12(:,:,i,j) = kernel12(i,j,:,:);
                    kernel22(:,:,i,j) = kernel22(i,j,:,:);
                end
            end
            
            % forward Euler
            Vx_new=zeros(p,p); Vy_new=zeros(p,p);
            for i = 1:p
                for j = 1:p
                    Vx_new(i,j) = Vx(ind_y(i),ind_x(j)) - RBM_num*dt*sum(sum(weight(ind_y,ind_x).*(kernel11(:,:,i,j).*(Fx(ind_y(i),ind_x(j))-Fx(ind_y,ind_x))+kernel12(:,:,i,j).*(Fy(ind_y(i),ind_x(j))-Fy(ind_y,ind_x)))));
                    Vy_new(i,j) = Vy(ind_y(i),ind_x(j)) - RBM_num*dt*sum(sum(weight(ind_y,ind_x).*(kernel12(:,:,i,j).*(Fx(ind_y(i),ind_x(j))-Fx(ind_y,ind_x))+kernel22(:,:,i,j).*(Fy(ind_y(i),ind_x(j))-Fy(ind_y,ind_x)))));
                end
            end
            Vx_temp(ind_y,ind_x) = Vx_new;
            Vy_temp(ind_y,ind_x) = Vy_new;
            
        end
    end
    toc
    
    
    % energy conservation trick
    variance_temp = sum(sum(weight.*((Vx_temp-target_momentum_x).^2+(Vy_temp-target_momentum_y).^2)));
    Vx = (Vx_temp - target_momentum_x)*sqrt(target_variance/variance_temp) + target_momentum_x;
    Vy = (Vy_temp - target_momentum_y)*sqrt(target_variance/variance_temp) + target_momentum_y;
    
    
    % update particle solution at the square centers
    % compute sum_k w_k psi (vc-vk) for next iteration
    f = zeros(Nv,Nv);
    for i = 1:Nv
        for j = 1:Nv
            f(i,j) = sum(sum(weight.*mollifier2d(Vcx(i,j)-Vx,Vcy(i,j)-Vy,eps)));
        end
    end
    f_slice_x(iit,:) = f(round(Nv/2),:); 
    f_slice_y(iit,:) = f(:,round(Nv/2));
    
    
    
    % conserved quantities
    group_velocity_x(iit) = sum(sum(weight.*Vx)); group_velocity_y(iit) = sum(sum(weight.*Vy));
    energy(iit) = sum(sum(weight.*(Vx.^2+Vy.^2)));
    entropy(iit) = sum(sum(f.*log(f)))*h^d;
    maxparticle_x(iit) = max(max(abs(Vx))); maxparticle_y(iit) = max(max(abs(Vy)));
    num = histcounts2(Vx,Vy,'XBinLimits',[-v_max,v_max],'YBinLimits',[-v_max,v_max]);
    out_of_region(iit) = 1-sum(sum(num))/N;
    
%     % exact sol at square ceters
%     K = 1-exp(-tt(iit)/8)/2;
%     f_exact = 1/2/pi/K*exp(-norm_v_square/2/K).*((2*K-1)/K+(1-K)/2/K^2*norm_v_square);
%     L2error(iit) = sqrt(sum(sum((f_l-f_exact).^2))*h^d)/sqrt(sum(sum((f_exact).^2))*h^d);
%     L2error(iit) = sqrt(sum(sum((f-f_Landau(1:times:end,1:times:end,iit)).^2))*h^d)/sqrt(sum(sum((f_Landau(1:times:end,1:times:end,iit)).^2))*h^d);
    
    if mod(iit-1,10)==0
        subplot(1,2,1)
        histogram2(Vx,Vy)
        title(tt(iit))
        subplot(1,2,2)
        surf(Vcx,Vcy,f)
        xlabel('Vx'), ylabel('Vy')
        xlim([-v_max,v_max]), ylim([-v_max,v_max])
        view([20 80 45])
        title(tt(iit))
        disp(sum(sum(f))*h^2)
        disp([max(max(abs(Vx))),max(max(abs(Vy)))])
        disp(['out of region=',num2str(out_of_region(iit)*100),'%'])
        pause(0.05)
    end
    
    
end
time = toc;

f_num = f;
% error
% relative_L2 = L2error(Nt);
% relative_L1 = sum(sum(abs(f_num-f_Landau(1:times:end,1:times:end,end))))/sum(sum(abs(f_Landau(1:times:end,1:times:end,end))));
% relative_Linf = max(max(abs(f_num-f_Landau(1:times:end,1:times:end,end))))/max(max(abs(f_Landau(1:times:end,1:times:end,end))));
% relative_L1 = sum(sum(abs(f_num-f_exact)))/sum(sum(abs(f_exact)));
% relative_Linf = max(max(abs(f_num-f_exact)))/max(max(abs(f_exact)));

figure(2)
subplot(2,2,1)
plot(tt,energy)
ylabel('energy')
subplot(2,2,2)
plot(tt,group_velocity_x,tt,group_velocity_y)
ylabel('momentum')
legend('x dim','y dim')
subplot(2,2,3)
semilogy(tt,entropy)
ylabel('entropy')
subplot(2,2,4)
semilogy(tt,L2error)
ylabel('relative L_2 error')

figure(3)
subplot(1,2,1)
plot(tt,maxparticle_x)
subplot(1,2,2)
plot(tt,out_of_region*100)

figure(4)
for iit=1:Nt
    subplot(1,2,1)
    plot(v_x,f_slice_x(iit,:))
    subplot(1,2,2)
    plot(v_y,f_slice_y(iit,:))
    title(tt(iit))
    pause(0.02)
end


disp('------------------------------------------------')
disp(['n= ', num2str(Nv)])
disp(['CPU time= ',num2str(time)])
disp(['mass= ',num2str(sum(sum(weight)))])
disp(['group velocity = [',num2str(group_velocity_x(Nt)),',',num2str(group_velocity_y(Nt)),']'])
disp(['kinetic energy = ',num2str(energy(Nt))])
disp(['max particle position = [',num2str(maxparticle_x(Nt)),',',num2str(maxparticle_y(Nt)),']'])
% disp('relative error:')
% disp(['L_1 = ',num2str(relative_L1), '  L_2 = ',num2str(relative_L2),'  L_inf = ',num2str(relative_Linf)])


