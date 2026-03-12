% ---------------------------------------------------------------------------------------
% Name : SCSn_general.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate PO and PTD normalize scattering cross-section for paraboloid and spherical shapes for
% soft and hard boundary conditions
% ---------------------------------------------------------------------------------------
function [PO,PTDs,PTDh] = SCSn_general(ka,kl,w,Omega,Shape)

PRB = strcmp(Shape,'Paraboloid');
SPH = strcmp(Shape,'Spherical');
n = 1+(w+Omega)/pi;

if (PRB == 1) % for paraboloid surfaces
    kR = ka*tan(w); % kR = k*R where R is the curvature radius at rho=0;
elseif (SPH == 1) % for spherical surfaces
    kR = ka/cos(w); % kR = k*R where R is the curvature radius at rho=0;
end

if (w == pi/2)
    Ish0 = ka^2;
    Is = ka*(1i*ka+(1/n)*cot(pi/n)+(2/n)*sin(pi/n)/(cos(pi/n)-1));
    Ih = ka*(1i*ka+(1/n)*cot(pi/n)-(2/n)*sin(pi/n)/(cos(pi/n)-1));
else
    Ish0 = kR-ka*(tan(w)*exp(1i*2*kl));
    Is = kR-ka*((2/n)*sin(pi/n)*((1./(cos(pi/n)-1))-(1./(cos(pi/n)-cos(2*w./n)))).*exp(1i*2*kl));
    Ih = kR+ka*((2/n)*sin(pi/n)*((1./(cos(pi/n)-1))+(1./(cos(pi/n)-cos(2*w./n)))).*exp(1i*2*kl));
end

PO = abs(Ish0)^2/(ka^2); % Normalized scattering cross-section for PO approximation for SBC and HBC
PTDs = abs(Is)^2/(ka^2); % Normalized scattering cross-section for PTD approximation for SBC
PTDh = abs(Ih)^2/(ka^2); % Normalized scattering cross-section for PTD approximation for SBC
% ****************************************** End ******************************************
end
