% ---------------------------------------------------------------------------------------
% Name : FG.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate the spherical components of differential diffraction coefficients F1 and G1
% ---------------------------------------------------------------------------------------
function [F1_Vt,F1_phi,G1_Vt,G1_phi] = FG(angle,angle0,gamma0,Vtheta,alfa)
if (Vtheta == pi-gamma0)
[f1,g1] = fun_fg(angle,angle0,alfa);
F1_Vt = -f1*sin(Vtheta)/sin(gamma0); F1_phi = 0;
G1_Vt = (eps_x(angle0)-eps_x(alfa-angle0))*cot(gamma0);
G1_phi = -g1*sin(Vtheta)/sin(gamma0);
else
beta1 = acos(sin(gamma0)*sin(Vtheta)*cos(angle)-cos(gamma0)*cos(Vtheta)); % Eq.(7.75)
beta2 = acos(sin(gamma0)*sin(Vtheta)*cos(alfa-angle)-cos(gamma0)*cos(Vtheta)); % Eq.(7.95)
sgm1 = sigma12(beta1,gamma0); sgm2 = sigma12(beta2,gamma0);
Ut1 = (pi/(2*alfa*sin(gamma0).^2))*(cot(pi*(sgm1+angle0)/(2*alfa))-cot(pi*(sgm1-angle0)/(2*alfa))); % Eq.(7.85)
U01 = -eps_x(angle0)*sin(angle0)/(cos(beta1)-cos(gamma0).^2+sin(gamma0).^2*sin(angle0).^2); % Eq.(7.86)
Vt1 = (pi/(2*alfa*sin(gamma0).^2))*(cot(pi*(sgm1+angle0)/(2*alfa))+cot(pi*(sgm1-angle0)/(2*alfa))); % Eq.(7.87)
V01 = eps_x(angle0)/(cos(beta1)-cos(gamma0).^2+sin(gamma0).^2*cos(angle0).^2); % Eq.(7.88)
Ut2 = (pi/(2*alfa*sin(gamma0).^2))*(cot(pi*(sgm2+alfa-angle0)/(2*alfa))-cot(pi*(sgm2-alfa+angle0)/(2*alfa)));
U02 = -eps_x(alfa-angle0)*sin(alfa-angle0)/(cos(beta2)-cos(gamma0).^2+sin(gamma0).^2*cos(alfa-angle0).^2);
Vt2 = (pi/(2*alfa*sin(gamma0).^2))*(cot(pi*(sgm2+alfa-angle0)/(2*alfa))+cot(pi*(sgm2-alfa+angle0)/(2*alfa)));
V02 = eps_x(alfa-angle0)/(cos(beta2)-cos(gamma0).^2+sin(gamma0).^2*cos(alfa-angle0).^2);
U1 = Ut1-U01; V1 = Vt1-V01; % Eq.(7.83); Eq.(7.84)
U2 = Ut2-U02; V2 = Vt2-V02; % Eq.(7.93); Eq.(7.94)
if (sgm1 == angle0)
U1 = pi*cot(pi*angle0/alfa)/(2*alfa*sin(gamma0).^2)-cot(angle0)/(2*sin(gamma0).^2); % Eq.(7.106)
V1 = U1/sin(angle0); % Eq.(7.107)
end
if (sgm2 == alfa-angle0)
U2 = pi*cot(pi*(alfa-angle0)/alfa)/(2*alfa*sin(gamma0).^2)-cot(alfa-angle0)/(2*sin(gamma0).^2); % Eq.(7.108)
V2 = U2/sin(alfa-angle0); % Eq.(7.109)
end
Fs1 = -(U1+U2).*sin(gamma0).^2; % Eq.(7.91)
F1_Vt = Fs1; F1_phi = 0;
G1_Vt = (eps_x(angle0)-eps_x(alfa-angle0))*sin(Vtheta)*cos(gamma0)/(sin(gamma0).^2)...
+(sin(gamma0)*cos(Vtheta)*cos(angle)-cos(gamma0)*sin(Vtheta))*V1...
-(sin(gamma0)*cos(Vtheta)*cos(alfa-angle)-cos(gamma0)*sin(Vtheta))*V2;
G1_phi = -(V1*sin(angle)+V2*sin(alfa-angle))*sin(gamma0);
end
end
