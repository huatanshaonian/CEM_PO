% ---------------------------------------------------------------------------------------
% Name : electromagnetic_EEW_Section78.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate the spherical components of differential diffraction coefficients F1 and G1 in Eq.(7.137)
% ---------------------------------------------------------------------------------------
alfa = input('wedge angle [Deg] (pi < alfa < 2*pi) = ')*pi/180;
angle0 = input('incident angle [Deg] (0 < angle0 < alfa) = ')*pi/180;
gamma0 = input('gamma_0 angle [Deg] (0 < gamma0 < pi) = ')*pi/180;
Vtheta = input('vartheta angle [Deg] (0 < Vtheta < pi) = ')*pi/180;
angle = input('observation angle [Deg] (0 <= angle <= alfa) = ');
% ------------------------------------------------ change the angles degrees to radians
Vtheta = Vtheta*pi/180; angle = angle*pi/180;
% ------------------------------------------------ length of the inputs
l1 = length(alfa); l2 = length(angle0); l3 = length(gamma0); l4 = length(Vtheta); l5 = length(angle);
if (l4 > 1)
Ar = Vtheta; nA = l4; var = 1; % for Figure 7.10
elseif (l5 > 1)
Ar = angle; nA = l5; var = 2; % for Figure 7.9 and Figure 7.11
end
for m = 1:nA
if (var == 1)
fprintf(1, 'Calculating directivity pattern for Vtheta %5.1f : \n',rad2deg(Ar(m))); Vtheta = Ar(m);
if (angle == alfa/2+pi)
teta(m) = 2*pi-Vtheta; Ar(m) = teta(m);
elseif (angle == alfa/2)
teta(m) = Vtheta; Ar(m) = teta(m);
end
elseif (var == 2)
fprintf(1, 'Calculating directivity pattern for angle %5.1f : \n',rad2deg(Ar(m))); angle = Ar(m);
end
% spherical components of differential diffraction coefficients F1 and G1
[F1_Vt(m),F1_phi(m),G1_Vt(m),G1_phi(m)] = FG(angle,angle0,gamma0,Vtheta,alfa);
% calculating the quantities 10 log |F^(1)_theta,phi| and 10 log |G^(1)_theta,phi|
F1L(m) = 10*log10(abs(F1_Vt(m))); G11L(m) = 10*log10(abs(G1_Vt(m))); G12L(m) = 10*log10(abs(G1_phi(m)));
end
% ------------------------------------------------ END ------------------------------------------------
