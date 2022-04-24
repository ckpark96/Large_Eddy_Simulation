% AE4139 CFD3 Assignment 1
% by Changkyu Park 4646061
% Script requires uvw_physical.mat & uvw_fourier.mat files

%% Initial setup
% cmap = cmocean('haline'); % requires an additional Toolbox
cmap = hot;
% Provided parameters
N        = 192;
kin_visc = 0.0008;
% Import data
vel_field     = load('uvw_physical.mat');
fourier_mode  = load('uvw_fourier.mat');
% Velocity field - function of x
u  = vel_field.u;
v  = vel_field.v;
w  = vel_field.w;
% Corresponding Fourier modes (u-hat) - functions of xi
uk = fourier_mode.uk;
vk = fourier_mode.vk;
wk = fourier_mode.wk;
% Physical coordinates
x = linspace(0,2*pi*(1-1/N),N);
h = 2*pi*(1-1/N)/(N-1);
% Fourier wave numbers
xi = cat(2,0:N/2-1,-N/2:-1);
% Meshgrid of fourier modes
[XIx,XIy,XIz] = ndgrid(xi); % row-x, column-y, layer-z

%% Spectral differentiation
dudx_sp = uk.*XIx*1i;
dvdy_sp = vk.*XIy*1i;
dwdz_sp = wk.*XIz*1i;

%% Central difference scheme - 2nd order
% Sparse finite difference matrices
deriv        = diag(ones(1,N-1),1) + diag(-1*ones(1,N-1),-1);
deriv(1,end) = -1;
deriv(end,1) = 1;
deriv        = sparse(deriv);
% Derivative matrices - dudx, dvdy, dwdz
Eye      = speye(N);
deriv_X  = kron(kron(Eye,Eye),deriv);
deriv_Y  = kron(kron(Eye,deriv),Eye);
deriv_Z  = kron(kron(deriv,Eye),Eye);
% Flatten 3D matrices to 1D
flatU = reshape(u,1,[]);
flatV = reshape(v,1,[]);
flatW = reshape(w,1,[]);
% Calculate derivatives
dudx_fd_1d = (deriv_X*flatU'/2/h);
dvdy_fd_1d = (deriv_Y*flatV'/2/h);
dwdz_fd_1d = (deriv_Z*flatW'/2/h);
% Unflatten 1D to 3D matrices
dudx_fd = reshape(dudx_fd_1d,N,N,N);
dvdy_fd = reshape(dvdy_fd_1d,N,N,N);
dwdz_fd = reshape(dwdz_fd_1d,N,N,N);

%% Root-mean-squared 
div_spec  = real(dudx_sp + dvdy_sp + dwdz_sp);
rms_spec  = sqrt(sum(div_spec.^2,'all'));
div_fd    = dudx_fd + dvdy_fd + dwdz_fd;
rms_fd    = sqrt(sum(div_fd.^2,'all')/N/N/N);

%% Q-criterion
dudy_sp = uk.*XIy*1i;
dudz_sp = uk.*XIz*1i;
dvdx_sp = vk.*XIx*1i;
dvdz_sp = vk.*XIz*1i ;
dwdx_sp = wk.*XIx*1i;
dwdy_sp = wk.*XIy*1i;
dudx    = real(ifftn(dudx_sp));
dudy    = real(ifftn(dudy_sp));
dudz    = real(ifftn(dudz_sp));
dvdx    = real(ifftn(dvdx_sp));
dvdy    = real(ifftn(dvdy_sp));
dvdz    = real(ifftn(dvdz_sp));
dwdx    = real(ifftn(dwdx_sp));
dwdy    = real(ifftn(dwdy_sp));
dwdz    = real(ifftn(dwdz_sp));
Q       = -0.5*(dudx.*dudx+dudy.*dvdx+dudz.*dwdx+...
                dvdx.*dudy+dvdy.*dvdy+dvdz.*dwdy+...
                dwdx.*dudz+dwdy.*dvdz+dwdz.*dwdz)*N^6;
            
%% Iso-surfaces
isoQ = [1800,2000];
isosurface(Q,isoQ(1));
% hold on
% isosurface(Q,isoQ(2));

%% Box-averaged TKE
k_phy = 1/N/N/N*sum(0.5*(u.^2+v.^2+w.^2),'all');
k_sp  = 1/2*sum((uk.*conj(uk)+vk.*conj(vk)+wk.*conj(wk)),'all');

%% Dissipation rate
eps = kin_visc*sum((XIx.^2+XIy.^2+XIz.^2).*(uk.*conj(uk)+vk.*conj(vk)+wk.*conj(wk)),'all');

%% 3D KE density spectrum
wave_no_mag = sqrt(XIx.^2+XIy.^2+XIz.^2);
E           = 1/2*(uk.*conj(uk)+vk.*conj(vk)+wk.*conj(wk));
En_gather   = [];
nmax        = round(max(max(max(wave_no_mag))));
nrange      = 1:nmax; % equals to xi_n
for n = nrange
    En           = sum(E(n-0.5<=wave_no_mag & wave_no_mag<n+0.5),'all');
    En_gather(n) = En;
end
slope = nrange.^(-5/3)*10^0.5; % slope to visualise -5/3 rule
% Plot
figure(1)
hold off
loglog(nrange,En_gather,'k-');
hold on
plot(nrange,slope,'r-');
xlim([1,200])
grid on
xlabel('\xi_n', 'FontSize', 16);
ylabel('\it E', 'FontSize', 16);
legend({'E','slope = -5/3'}, 'FontSize', 12);

%% Dissipation spectrum
D = 2*kin_visc*nrange.^2.*En_gather;
% Plot
figure(2)
hold off
loglog(nrange,D,'k-');
xlim([1,200])
grid on
xlabel('\xi_n', 'FontSize', 16);
ylabel('\it D', 'FontSize', 16);

%% 3D spectrum - Boxed averaged TKE and dissipation rate
k_3d   = sum(En_gather);
eps_3d = sum(D);

%% Kolmogorov scaling and constant
Ctilde     = En_gather.*eps^(-2/3).*nrange.^(5/3);
nrange_IR = nrange(4:55); % Range is read off the plot
Ctilde_IR = Ctilde(4:55);
Ck        = mean(Ctilde_IR);

%% Reynolds number and corresponding length calculation
k         = k_sp; % k is same for all methods, thus just a random one has been chosen
% Integral length scale
Re_L      = k^2/eps/kin_visc;
L         = k^(3/2)/eps;
% Taylor micro scale
lambda_g  = sqrt(10*kin_visc*k/eps);
Re_lambda = lambda_g*sqrt(2/3*k)/kin_visc;

%% Filtration
NLES = 24;
hLES = 2*pi*(1-1/NLES)/(NLES-1); % Filter width
xi_N = pi/hLES; % Nyquist wavenumber
% Filtered modes
filter1 = 2*xi_N/pi./wave_no_mag.*sin(pi*wave_no_mag/2/xi_N);
filter1(1,1,1) = 1; % Redefine the limit of wave number magnitude = 0 as 1 
filter2 = wave_no_mag<=xi_N; % True=1, False=0
ukLES   = uk.*filter1.*filter2; 
vkLES   = vk.*filter1.*filter2;
wkLES   = wk.*filter1.*filter2;
% 3D kinetic energy spectrum
ELES           = 1/2*(ukLES.*conj(ukLES)+vkLES.*conj(vkLES)+wkLES.*conj(wkLES));
ELESn_gather   = [];
% Loop through values of n to sum values that fits in the range
for n = nrange
    ELESn           = sum(ELES(n-0.5<=wave_no_mag & wave_no_mag<n+0.5),'all');
    ELESn_gather(n) = ELESn;
end
% Plot for comparison with non-filtered
figure(3)
hold off
loglog(nrange,ELESn_gather,'-ok');
hold on
loglog(nrange,En_gather,'-xk');
xlim([1,200])
grid on
xlabel('\xi_n', 'FontSize', 16);
ylabel('\it E', 'FontSize', 16);
legend({'LES','DNS'}, 'FontSize', 12);
% Dissipation spectrum
DLES = 2*kin_visc*nrange.^2.*ELESn_gather;
% Plot
figure(4)
hold off
loglog(nrange,DLES,'-ok');
hold on
loglog(nrange,D,'-xk');
xlim([1,200])
grid on
xlabel('\xi_n', 'FontSize', 16);
ylabel('\it D', 'FontSize', 16);
legend({'LES','DNS'}, 'FontSize', 12);

%% Vorticity and velocity magnitude
% Vorticity
vort_z       = (dvdx-dudy)*N^3;
dvdx_spLES   = vkLES.*XIx*1i;
dudy_spLES   = ukLES.*XIy*1i;
vort_zLES    = real(ifftn(dvdx_spLES-dudy_spLES))*N^3;
% Velocity
u_mag        = sqrt(u.^2+v.^2+w.^2);
uLES         = real(ifftn(ukLES))*N^3;
vLES         = real(ifftn(vkLES))*N^3;
wLES         = real(ifftn(wkLES))*N^3;
u_magLES     = sqrt(uLES.^2+vLES.^2+wLES.^2);
[X,Y]        = ndgrid(x);
% Plot vorticity
plane      = N/2; % Chosen x-y plane
contourlvl = 10; % Number of countourlevels
font1      = 13; % Font size defined for convenience
% Plots
figure(5)
[C1,h1] = contour(X,Y,vort_z(:,:,plane),contourlvl);
colormap(cmap)
c = colorbar;
c.FontSize = 11;
xlabel('X', 'FontSize', font1);
ylabel('Y', 'FontSize', font1);
title('DNS', 'FontSize', font1);
figure(6)
[C2,h2] = contour(X,Y,vort_zLES(:,:,plane),contourlvl);
colormap(cmap)
c = colorbar;
c.FontSize = 11;
xlabel('X', 'FontSize', font1);
ylabel('Y', 'FontSize', font1);
title('LES', 'FontSize', font1);
% Plot velocity
figure(7)
[C1,h1] = contour(X,Y,u_mag(:,:,plane),contourlvl);
colormap(cmap)
c = colorbar;
c.FontSize = 11;
xlabel('X', 'FontSize', font1);
ylabel('Y', 'FontSize', font1);
title('DNS', 'FontSize', font1);
figure(8)
[C2,h2] = contour(X,Y,u_magLES(:,:,plane),contourlvl);
colormap(cmap)
c = colorbar;
c.FontSize = 11;
xlabel('X', 'FontSize', font1);
ylabel('Y', 'FontSize', font1);
title('LES', 'FontSize', font1);

% Exact SGS stress tensor - product in physical space, filtration in spectral space
SGS11 = real(ifftn(fftn(u.*u)/N^3.*filter1.*filter2)*N^3) - real(ifftn(ukLES))*N^3.*real(ifftn(ukLES)*N^3);
SGS22 = real(ifftn(fftn(v.*v)/N^3.*filter1.*filter2)*N^3) - real(ifftn(vkLES))*N^3.*real(ifftn(vkLES)*N^3);
SGS33 = real(ifftn(fftn(w.*w)/N^3.*filter1.*filter2)*N^3) - real(ifftn(wkLES))*N^3.*real(ifftn(wkLES)*N^3);
SGS12 = real(ifftn(fftn(u.*v)/N^3.*filter1.*filter2)*N^3) - real(ifftn(ukLES))*N^3.*real(ifftn(vkLES)*N^3);
SGS13 = real(ifftn(fftn(u.*w)/N^3.*filter1.*filter2)*N^3) - real(ifftn(ukLES))*N^3.*real(ifftn(wkLES)*N^3);
SGS23 = real(ifftn(fftn(v.*w)/N^3.*filter1.*filter2)*N^3) - real(ifftn(vkLES))*N^3.*real(ifftn(wkLES)*N^3);
font2 = 13; % font size defined for convenience
% Plots
figure(9)
subplot(3,2,1);
[s1,g1] = contour(X,Y,SGS11(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{11}', 'FontSize', font2);
subplot(3,2,2);
[s2,g2] = contour(X,Y,SGS12(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{12}(\tau_{21})', 'FontSize', font2);
subplot(3,2,3);
[s3,g3] = contour(X,Y,SGS13(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{13}(\tau_{31})', 'FontSize', font2);
subplot(3,2,4);
[s4,g4] = contour(X,Y,SGS22(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{22}', 'FontSize', font2);
subplot(3,2,5);
[s5,g5] = contour(X,Y,SGS23(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{23}(\tau_{32})', 'FontSize', font2);
subplot(3,2,6);
[s6,g6] = contour(X,Y,SGS33(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{33}', 'FontSize', font2);

%% Smagorinsky - Eddy viscosity
Cs   = 0.17;
S11  = real(ifftn(ukLES.*XIx.*1i)).*N^3;
S22  = real(ifftn(vkLES.*XIy.*1i)).*N^3;
S33  = real(ifftn(wkLES.*XIz.*1i)).*N^3;
S12a = real(ifftn(ukLES.*XIy.*1i)).*N^3;
S12b = real(ifftn(vkLES.*XIx.*1i)).*N^3;
S12  = 0.5.*(S12a+S12b); % Same as S21
S21  = S12;
S13a = real(ifftn(ukLES.*XIz.*1i)).*N^3;
S13b = real(ifftn(wkLES.*XIx.*1i)).*N^3;
S13  = 0.5.*(S13a+S13b); % Same as S31
S31  = S13;
S23a = real(ifftn(vkLES.*XIz.*1i)).*N^3;
S23b = real(ifftn(wkLES.*XIy.*1i)).*N^3;
S23  = 0.5.*(S23a+S23b); % Same as S32
S32  = S23;
Sijnorm  = sqrt(2.*S11.^2+2.*S22.^2+2.*S33.^2+2.*S12.^2+2.*S13.^2+2.*S31.^2+2.*S23.^2+2.*S32.^2);
% Subgrid scale viscosity
vSGS  = ((Cs*hLES)^2).*Sijnorm;
% Plot
figure(10)
[mu1,mu2] = contour(X,Y,vSGS(:,:,plane),contourlvl);
colormap(cmap);
xlabel('X', 'FontSize', font1);
ylabel('Y', 'FontSize', font1);
c = colorbar;
c.FontSize = 11;
title('\nu_{SGS}', 'FontSize', font1);
% Smagorinsky subgrid scale stress tensor
smaSGS11 = -2*vSGS.*S11;
smaSGS22 = -2*vSGS.*S22;
smaSGS33 = -2*vSGS.*S33;
smaSGS12 = -2*vSGS.*S12;
smaSGS13 = -2*vSGS.*S13;
smaSGS23 = -2*vSGS.*S23;
% Plots
figure(11)
subplot(3,2,1);
[ss1,gg1] = contour(X,Y,smaSGS11(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{11}', 'FontSize', font2);
subplot(3,2,2);
[ss2,gg2] = contour(X,Y,smaSGS12(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{12}(\tau_{21})', 'FontSize', font2);
subplot(3,2,3);
[ss3,gg3] = contour(X,Y,smaSGS13(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{13}(\tau_{31})', 'FontSize', font2);
subplot(3,2,4);
[ss4,gg4] = contour(X,Y,smaSGS22(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{22}', 'FontSize', font2);
subplot(3,2,5);
[ss5,gg5] = contour(X,Y,smaSGS23(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{23}(\tau_{32})', 'FontSize', font2);
subplot(3,2,6);
[ss6,gg6] = contour(X,Y,smaSGS33(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{33}', 'FontSize', font2);

%% Bardina model - Galilei invariant
B11a = ifftn(fftn(uLES.*uLES)/N^3.*filter1.*filter2)*N^3;
B11b = ifftn(ukLES.*filter1.*filter2)*N^3.*ifftn(ukLES.*filter1.*filter2)*N^3;
B11  = real(B11a-B11b);
B22a = ifftn(fftn(vLES.*vLES)/N^3.*filter1.*filter2)*N^3;
B22b = ifftn(vkLES.*filter1.*filter2)*N^3.*ifftn(vkLES.*filter1.*filter2)*N^3;
B22  = real(B22a-B22b);
B33a = ifftn(fftn(wLES.*wLES)/N^3.*filter1.*filter2)*N^3;
B33b = ifftn(wkLES.*filter1.*filter2)*N^3.*ifftn(wkLES.*filter1.*filter2)*N^3;
B33  = real(B33a-B33b);
B12a = ifftn(fftn(uLES.*vLES)/N^3.*filter1.*filter2)*N^3;
B12b = ifftn(ukLES.*filter1.*filter2)*N^3.*ifftn(vkLES.*filter1.*filter2)*N^3;
B12  = real(B12a-B12b);
B13a = ifftn(fftn(uLES.*wLES)/N^3.*filter1.*filter2)*N^3;
B13b = ifftn(ukLES.*filter1.*filter2)*N^3.*ifftn(wkLES.*filter1.*filter2)*N^3;
B13  = real(B13a-B13b);
B23a = ifftn(fftn(vLES.*wLES)/N^3.*filter1.*filter2)*N^3;
B23b = ifftn(vkLES.*filter1.*filter2)*N^3.*ifftn(wkLES.*filter1.*filter2)*N^3;
B23  = real(B13a-B13b);
% Plots
figure(12)
subplot(3,2,1);
[b1,B1] = contour(X,Y,B11(:,:,plane),contourlvl);
% set(B1,'LineColor','none')
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{11}', 'FontSize', font2);
subplot(3,2,2);
[b2,B2] = contour(X,Y,B12(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{12}(\tau_{21})', 'FontSize', font2);
subplot(3,2,3);
[b3,B3] = contour(X,Y,B13(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{13}(\tau_{31})', 'FontSize', font2);
subplot(3,2,4);
[b4,B4] = contour(X,Y,B22(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{22}', 'FontSize', font2);
subplot(3,2,5);
[b5,B5] = contour(X,Y,B23(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{23}(\tau_{32})', 'FontSize', font2);
subplot(3,2,6);
[b6,B6] = contour(X,Y,B33(:,:,plane),contourlvl);
xlabel('X', 'FontSize', font2);
ylabel('Y', 'FontSize', font2);
colormap(cmap)
colorbar
title('\tau_{33}', 'FontSize', font2);
