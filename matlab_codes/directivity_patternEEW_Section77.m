% ---------------------------------------------------------------------------------------
% Name : directivity_patternEEW_Section77.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate directivity patterns of EEWs and 10log|F^(1)_s,h|
% ---------------------------------------------------------------------------------------
alfa = input('wedge angle [Deg] (pi < alfa <= 2*pi) = ')*pi/180;
angle0 = input('incident angle [Deg] (0 < angle0 < alfa) = ')*pi/180;
gamma0 = input('gamma_0 angle [Deg] (0 < gamma0 < pi) = ')*pi/180;
Vtheta = input('vartheta angle [Deg] (0 < Vtheta < pi) = ')*pi/180;
angle = input('observation angle [Deg] (0 <= angle <= alfa) = '); 
% ------------------------------------------------ change the angles degrees to radians
l1 = length(alfa); l2 = length(angle0); l3 = length(gamma0); l4 = length(Vtheta); l5 = length(angle);
% ------------------------------------------------ length of the inputs
if (l4 > 1)
Ar = Vtheta; nA = l4; var = 1; % for Figure 7.7
elseif (l5 > 1)
Ar = angle*pi/180; nA = l5; var = 2; % for Figure 7.6 and Figure 7.8
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
[Fs1(m),Fh1(m)] = Fsh(angle,angle0,gamma0,Vtheta,alfa); % directivity patterns of EEWs
% calculating the quantities 10log|F^(1)_s| and 10log|F^(1)_h|
Fs(m) = 10*log10(abs(Fs1(m))); Fh(m) = 10*log10(abs(Fh1(m)));
end
% ------------------------------------------------ END ------------------------------------------------
