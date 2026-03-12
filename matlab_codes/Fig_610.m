% ---------------------------------------------------------------------------------------
% Name : Fig_610.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate the PO and PTD backscattering fields at a cone vs cone length l
% ---------------------------------------------------------------------------------------
w = deg2rad(45); omeg = deg2rad(90);
KLL = 10:0.01:30;
[dd, nKLL] = size(KLL);

for m = 1:nKLL
    kl = KLL(m); ka = kl;
    n = 1+(w+omeg)/pi;
    if (w == pi/2) % case w=pi/2
        Ish0 = ka;
        Is = 1i*ka+(1/n)*cot(pi/n)+(2/n)*sin(pi/n)/(cos(pi/n)-1);
        Ih = -1i*ka-(1/n)*cot(pi/n)+(2/n)*sin(pi/n)/(cos(pi/n)-1);
    elseif (w == pi/2 && omeg == pi/2) % case w=pi/2 and omega = pi/2
        Ish0 = ka;
        Is = 1i*ka-1;
        Ih = 1i*ka+1;
    else
        Ish0 = 1i*tan(w)^2*(1-exp(1i*2*kl))/(2*ka)-tan(w)*exp(1i*2*kl);
        Is = 1i*tan(w)^2*(1-exp(1i*2*kl))/(2*ka)-(2*sin(pi/n)./n).*((1./(cos(pi/n)-1))-(1./(cos(pi/n)-cos(2*w./n)))).*exp(1i*2*kl);
        Ih = 1i*tan(w)^2*(1-exp(1i*2*kl))/(2*ka)+(2*sin(pi/n)./n).*((1./(cos(pi/n)-1))+(1./(cos(pi/n)-cos(2*w./n)))).*exp(1i*2*kl);
    end
    SCS_POsh(m) = abs(Ish0)^2; % normalized scattering cross-section for PO approximation for SBC and HBC
    SCS_PTDs(m) = abs(Is)^2;   % normalized scattering cross-section for PTD approximation for SBC
    SCS_PTDh(m) = abs(Ih)^2;   % normalized scattering cross-section for PTD approximation for HBC
end

figure
plot(KLL,10*log10(SCS_POsh),'k',KLL,10*log10(SCS_PTDs),'k:',KLL,10*log10(SCS_PTDh),'k--')
legend('PO','PTD SOFT','PTD HARD');
xlabel('kl'); ylabel('Normalized Scattering Cross-Section');
title('Figure 6.10');
