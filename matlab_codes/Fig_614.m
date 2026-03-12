% ---------------------------------------------------------------------------------------
% Name : Fig_614.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate the PO and PTD fields for conformal paraboloids
% ---------------------------------------------------------------------------------------
p = 3*pi*tan(deg2rad(14)); Omega = pi/2;
KLL = 6*pi:0.01:36;
[dd, nKLL] = size(KLL);

for m = 1:nKLL
    kl = KLL(m);
    w = acot(sqrt(2*kl/p));
    ka = 2*kl*tan(w);
    [PO,PTDs,PTDh] = SCSn_general(ka,kl,w,Omega,'Paraboloid');
    SCS_POsh(m) = PO; 
    SCS_PTDs(m) = PTDs; 
    SCS_PTDh(m) = PTDh;
end

figure
plot(KLL,10*log10(SCS_POsh),'k',KLL,10*log10(SCS_PTDs),'k:',KLL,10*log10(SCS_PTDh),'k--')
legend('PO','PTD SOFT','PTD HARD');
xlabel('kl'); ylabel('Normalized Scattering Cross-Section');
title('Figure 6.14');
