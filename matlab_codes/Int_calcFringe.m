% ---------------------------------------------------------------------------------------
% Name : Int_calcFringe.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate normalized fringe waves using (4.16), (4.17) for SSI and (4.18), (4.19) for DSI
% ---------------------------------------------------------------------------------------
function result = Int_calcFringe(alfa,kr,angle0,angle,S_H)
% S_H : Boundary condition: 'Soft' or 'Hard'
eps1 = 1e-12; y_old = 1e6; y1_old = 1e6; y2_old = 1e6; Mmax = 500;
eps = 1e-6; % 0<eps<<1 in radians
d1 = 0.001; n = alfa/pi;
psi1 = angle-angle0; psi2 = angle+angle0;
phi1 = -psi1; phi2 = 2*alfa-psi2;
C1 = exp(1i*(kr+pi/4))./(sqrt(2)*pi);
% %%----------------------------------------------- SSI ----------------------------------
if (angle0>=eps) && (angle0<=alfa-pi-eps) % Single side illumination
if (S_H == 'Soft') % Soft Boundary Conditions
sm = -d1:0.0001:d1; % -d1 and d1 are lower and upper limits of integration
ksim = -1i*sign(sm).*log(1+1i*sm.*sm+1i*abs(sm).*sqrt(sm.*sm-2*1i));
if (psi1<=pi+eps) && (psi1>=pi-eps)
AA = (ksim+psi1-pi);
XX = (1/12)*(1-(1/n)^2).*AA+(1/720)*(1-(1/n)^4)*(AA.^3); % series expansion
Fm1 = (XX-(1/(2*n)).*cot((ksim+psi1+pi)./(2*n))-(1/(2*n)).*cot((ksim+psi2-pi)./(2*n))...
+0.5*cot((ksim+psi2-pi)./2)+(1/(2*n)).*cot((ksim+psi2+pi)./(2*n)));
funm1 = exp(-kr*sm.*sm).*Fm1./cos(ksim./2); ym1 = trapz(sm,funm1); Int_m = C1*ym1;
for M=3:Mmax
s1 = -M:0.0001:-d1; % -M and -d1 are lower and upper limits of integration
ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))+0.5*cot((ksi1+psi2-pi)./2)+(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; % d1 and M are lower and upper limits of integration
ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))+0.5*cot((ksi2+psi2-pi)./2)+(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.16)
elseif (psi2<=pi+eps) && (psi2>=pi-eps)
BB = (ksim+psi2-pi); YY = (1/12)*(1-(1/n)^2).*BB+(1/720)*(1-(1/n)^4)*(BB.^3);
Fm2 = ((1/(2*n)).*cot((ksim+psi1-pi)./(2*n))-0.5*cot((ksim+psi1-pi)./2)-(1/(2*n)).*cot((ksim+psi1+pi)./(2*n))...
-YY+(1/(2*n)).*cot((ksim+psi2+pi)./(2*n)));
funm2 = exp(-kr*sm.*sm).*Fm2./cos(ksim./2); ym2 = trapz(sm,funm2); Int_m = C1*ym2;
for M=3:Mmax
s1 = -M:0.0001:-d1; ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))+0.5*cot((ksi1+psi2-pi)./2)+(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))+0.5*cot((ksi2+psi2-pi)./2)+(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.16)
else
for M=3:Mmax
s = -M:0.0001:M; ksi = -1i*sign(s).*log(1+1i*s.*s+1i*abs(s).*sqrt(s.*s-2*1i));
Fx = ((1/(2*n)).*cot((ksi+psi1-pi)./(2*n))-0.5*cot((ksi+psi1-pi)./2)-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi+psi2-pi)./(2*n))+0.5*cot((ksi+psi2-pi)./2)+(1/(2*n)).*cot((ksi+psi2+pi)./(2*n)));
fun = exp(-kr*s.*s).*Fx./cos(ksi./2); y(M) = trapz(s,fun); error = abs(y(M)-y_old);
y_old = y(M); if (error<eps1) Rx = y(M); break; else continue; end; end;
result = C1*Rx; % fringe wave given in (4.16)
end
elseif (S_H == 'Hard') % Hard Boundary Conditions
sm = -d1:0.0001:d1; ksi = -1i*sign(sm).*log(1+1i*sm.*sm+1i*abs(sm).*sqrt(sm.*sm-2*1i));
if (psi1<=pi+eps) && (psi1>=pi-eps)
AA = (ksi+psi1-pi); XX = (1/12)*(1-(1/n)^2).*AA+(1/720)*(1-(1/n)^4)*(AA.^3); % series expansion of the 1st and 2nd terms in (4.17)
Fm1 = (XX-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))+(1/(2*n)).*cot((ksi+psi2-pi)./(2*n))...
-0.5*cot((ksi+psi2-pi)./2)-(1/(2*n)).*cot((ksi+psi2+pi)./(2*n)));
funm1 = exp(-kr*sm.*sm).*Fm1./cos(ksi./2); ym1 = trapz(sm,funm1); Int_m = C1*ym1;
for M=3:Mmax
s1 = -M:0.0001:-d1; ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))-0.5*cot((ksi1+psi2-pi)./2)-(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))-0.5*cot((ksi2+psi2-pi)./2)-(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.17)
elseif (psi2<=pi+eps) && (psi2>=pi-eps)
BB = (ksi+psi2-pi); YY = (1/12)*(1-(1/n)^2).*BB+(1/720)*(1-(1/n)^4)*(BB.^3); % series expansion of the 3rd and 4th terms in (4.17)
Fm2 = ((1/(2*n)).*cot((ksi+psi1-pi)./(2*n))-0.5*cot((ksi+psi1-pi)./2)-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))...
+YY-(1/(2*n)).*cot((ksi+psi2+pi)./(2*n)));
funm2 = exp(-kr*sm.*sm).*Fm2./cos(ksi./2); ym2 = trapz(sm,funm2); Int_m = C1*ym2;
for M=3:Mmax
s1 = -M:0.0001:-d1; ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))-0.5*cot((ksi1+psi2-pi)./2)-(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))-0.5*cot((ksi2+psi2-pi)./2)-(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.17)
else
for M=3:Mmax
s = -M:0.0001:M; ksi = -1i*sign(s).*log(1+1i*s.*s+1i*abs(s).*sqrt(s.*s-2*1i));
Fx = ((1/(2*n)).*cot((ksi+psi1-pi)./(2*n))-0.5*cot((ksi+psi1-pi)./2)-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi+psi2-pi)./(2*n))-0.5*cot((ksi+psi2-pi)./2)-(1/(2*n)).*cot((ksi+psi2+pi)./(2*n)));
fun = exp(-kr*s.*s).*Fx./cos(ksi./2); y(M) = trapz(s,fun); error = abs(y(M)-y_old);
y_old = y(M); if (error<eps1) Rx = y(M); break; else continue; end; end;
result = C1*Rx; % fringe wave given in (4.17)
end; end;
% %%----------------------------------------------- DSI ----------------------------------
elseif (angle0>=alfa-pi+eps) && (angle0<=pi-eps) % Double side illumination
if (S_H == 'Soft') % Soft Boundary Conditions
sm = -d1:0.0001:d1; ksim = -1i*sign(sm).*log(1+1i*sm.*sm+1i*abs(sm).*sqrt(sm.*sm-2*1i));
if (psi1<=pi+eps) && (psi1>=pi-eps)
AA = (ksim+psi1-pi); XX = (1/12)*(1-(1/n)^2).*AA+(1/720)*(1-(1/n)^4)*(AA.^3); % series expansion of the 1st and 2nd terms in (4.18)
Fm1 = (XX-(1/(2*n)).*cot((ksim+psi1+pi)./(2*n))-(1/(2*n)).*cot((ksim+psi2-pi)./(2*n))...
+0.5*cot((ksim+psi2-pi)./2)+(1/(2*n)).*cot((ksim+psi2+pi)./(2*n))-(1/(2*n)).*cot((ksim+phi1+pi)./(2*n))...
+0.5*cot((ksim+phi1+pi)./2)+(1/(2*n)).*cot((ksim+phi2-pi)./(2*n))+0.5*cot((ksim+phi2-pi)./2)...
+(1/(2*n)).*cot((ksim+phi2+pi)./(2*n)));
funm1 = exp(-kr*sm.*sm).*Fm1./cos(ksim./2); ym1 = trapz(sm,funm1); Int_m = C1*ym1;
for M=3:Mmax
s1 = -M:0.0001:-d1; ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))+0.5*cot((ksi1+psi2-pi)./2)+(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n))...
-(1/(2*n)).*cot((ksi1+phi1+pi)./(2*n))+0.5*cot((ksi1+phi1+pi)./2)+(1/(2*n)).*cot((ksi1+phi2-pi)./(2*n))...
+0.5*cot((ksi1+phi2-pi)./2)+(1/(2*n)).*cot((ksi1+phi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))+0.5*cot((ksi2+psi2-pi)./2)+(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n))...
-(1/(2*n)).*cot((ksi2+phi1+pi)./(2*n))+0.5*cot((ksi2+phi1+pi)./2)+(1/(2*n)).*cot((ksi2+phi2-pi)./(2*n))...
+0.5*cot((ksi2+phi2-pi)./2)+(1/(2*n)).*cot((ksi2+phi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.18)
elseif (psi2<=pi+eps) && (psi2>=pi-eps)
BB = (ksim+psi2-pi); YY = (1/12)*(1-(1/n)^2).*BB+(1/720)*(1-(1/n)^4)*(BB.^3); % series expansion of the 3rd and 4th terms in (4.18)
Fm2 = ((1/(2*n)).*cot((ksim+psi1-pi)./(2*n))-0.5*cot((ksim+psi1-pi)./2)-(1/(2*n)).*cot((ksim+psi1+pi)./(2*n))...
-YY+(1/(2*n)).*cot((ksim+psi2+pi)./(2*n))-(1/(2*n)).*cot((ksim+phi1+pi)./(2*n))...
+0.5*cot((ksim+phi1+pi)./2)+(1/(2*n)).*cot((ksim+phi2-pi)./(2*n))+0.5*cot((ksim+phi2-pi)./2)...
+(1/(2*n)).*cot((ksim+phi2+pi)./(2*n)));
funm2 = exp(-kr*sm.*sm).*Fm2./cos(ksim./2); ym2 = trapz(sm,funm2); Int_m = C1*ym2;
for M=3:Mmax
s1 = -M:0.0001:-d1; ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))+0.5*cot((ksi1+psi2-pi)./2)+(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n))...
-(1/(2*n)).*cot((ksi1+phi1+pi)./(2*n))+0.5*cot((ksi1+phi1+pi)./2)+(1/(2*n)).*cot((ksi1+phi2-pi)./(2*n))...
+0.5*cot((ksi1+phi2-pi)./2)+(1/(2*n)).*cot((ksi1+phi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))+0.5*cot((ksi2+psi2-pi)./2)+(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n))...
-(1/(2*n)).*cot((ksi2+phi1+pi)./(2*n))+0.5*cot((ksi2+phi1+pi)./2)+(1/(2*n)).*cot((ksi2+phi2-pi)./(2*n))...
+0.5*cot((ksi2+phi2-pi)./2)+(1/(2*n)).*cot((ksi2+phi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.18)
else
for M=3:Mmax
s = -M:0.0001:M; ksi = -1i*sign(s).*log(1+1i*s.*s+1i*abs(s).*sqrt(s.*s-2*1i));
Fx = ((1/(2*n)).*cot((ksi+psi1-pi)./(2*n))-0.5*cot((ksi+psi1-pi)./2)-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))...
-(1/(2*n)).*cot((ksi+psi2-pi)./(2*n))+0.5*cot((ksi+psi2-pi)./2)+(1/(2*n)).*cot((ksi+psi2+pi)./(2*n))...
-(1/(2*n)).*cot((ksi+phi1+pi)./(2*n))+0.5*cot((ksi+phi1+pi)./2)+(1/(2*n)).*cot((ksi+phi2-pi)./(2*n))...
+0.5*cot((ksi+phi2-pi)./2)+(1/(2*n)).*cot((ksi+phi2+pi)./(2*n)));
fun = exp(-kr*s.*s).*Fx./cos(ksi./2); y(M) = trapz(s,fun); error = abs(y(M)-y_old);
y_old = y(M); if (error<eps1) Rx = y(M); break; else continue; end; end;
result = C1*Rx; % fringe wave given in (4.18)
end
elseif (S_H == 'Hard') % Hard Boundary Conditions
sm = -d1:0.0001:d1; ksi = -1i*sign(sm).*log(1+1i*sm.*sm+1i*abs(sm).*sqrt(sm.*sm-2*1i));
if (psi1<=pi+eps) && (psi1>=pi-eps)
AA = (ksi+psi1-pi); XX = (1/12)*(1-(1/n)^2).*AA+(1/720)*(1-(1/n)^4)*(AA.^3); % series expansion of the 1st and 2nd terms in (4.19)
Fm1 = (XX-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))+(1/(2*n)).*cot((ksi+psi2-pi)./(2*n))...
-0.5*cot((ksi+psi2-pi)./2)-(1/(2*n)).*cot((ksi+psi2+pi)./(2*n))+(1/(2*n)).*cot((ksi+phi1+pi)./(2*n))...
-0.5*cot((ksi+phi1+pi)./2)-(1/(2*n)).*cot((ksi+phi2-pi)./(2*n))-0.5*cot((ksi+phi2-pi)./2)...
+(1/(2*n)).*cot((ksi+phi2+pi)./(2*n)));
funm1 = exp(-kr*sm.*sm).*Fm1./cos(ksi./2); ym1 = trapz(sm,funm1); Int_m = C1*ym1;
for M=3:Mmax
s1 = -M:0.0001:-d1; ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))-0.5*cot((ksi1+psi2-pi)./2)-(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n))...
+(1/(2*n)).*cot((ksi1+phi1+pi)./(2*n))-0.5*cot((ksi1+phi1+pi)./2)-(1/(2*n)).*cot((ksi1+phi2-pi)./(2*n))...
-0.5*cot((ksi1+phi2-pi)./2)+(1/(2*n)).*cot((ksi1+phi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))-0.5*cot((ksi2+psi2-pi)./2)-(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n))...
+(1/(2*n)).*cot((ksi2+phi1+pi)./(2*n))-0.5*cot((ksi2+phi1+pi)./2)-(1/(2*n)).*cot((ksi2+phi2-pi)./(2*n))...
-0.5*cot((ksi2+phi2-pi)./2)+(1/(2*n)).*cot((ksi2+phi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.19)
elseif (psi2<=pi+eps) && (psi2>=pi-eps)
BB = (ksi+psi2-pi); YY = (1/12)*(1-(1/n)^2).*BB+(1/720)*(1-(1/n)^4)*(BB.^3); % series expansion of the 3rd and 4th terms in (4.19)
Fm2 = ((1/(2*n)).*cot((ksi+psi1-pi)./(2*n))-0.5*cot((ksi+psi1-pi)./2)-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))...
+YY-(1/(2*n)).*cot((ksi+psi2+pi)./(2*n))+(1/(2*n)).*cot((ksi+phi1+pi)./(2*n))...
-0.5*cot((ksi+phi1+pi)./2)-(1/(2*n)).*cot((ksi+phi2-pi)./(2*n))-0.5*cot((ksi+phi2-pi)./2)...
+(1/(2*n)).*cot((ksi+phi2+pi)./(2*n)));
funm2 = exp(-kr*sm.*sm).*Fm2./cos(ksi./2); ym2 = trapz(sm,funm2); Int_m = C1*ym2;
for M=3:Mmax
s1 = -M:0.0001:-d1; ksi1 = -1i*sign(s1).*log(1+1i*s1.*s1+1i*abs(s1).*sqrt(s1.*s1-2*1i));
F1 = ((1/(2*n)).*cot((ksi1+psi1-pi)./(2*n))-0.5*cot((ksi1+psi1-pi)./2)-(1/(2*n)).*cot((ksi1+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi1+psi2-pi)./(2*n))-0.5*cot((ksi1+psi2-pi)./2)-(1/(2*n)).*cot((ksi1+psi2+pi)./(2*n))...
+(1/(2*n)).*cot((ksi1+phi1+pi)./(2*n))-0.5*cot((ksi1+phi1+pi)./2)-(1/(2*n)).*cot((ksi1+phi2-pi)./(2*n))...
-0.5*cot((ksi1+phi2-pi)./2)+(1/(2*n)).*cot((ksi1+phi2+pi)./(2*n)));
fun1 = exp(-kr*s1.*s1).*F1./cos(ksi1./2); y1(M) = trapz(s1,fun1); error1 = abs(y1(M)-y1_old);
y1_old = y1(M); if (error1<eps1) R1 = y1(M); break; else continue; end; end; I1 = C1*R1;
for M=3:Mmax
s2 = d1:0.0001:M; ksi2 = -1i*sign(s2).*log(1+1i*s2.*s2+1i*abs(s2).*sqrt(s2.*s2-2*1i));
F2 = ((1/(2*n)).*cot((ksi2+psi1-pi)./(2*n))-0.5*cot((ksi2+psi1-pi)./2)-(1/(2*n)).*cot((ksi2+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi2+psi2-pi)./(2*n))-0.5*cot((ksi2+psi2-pi)./2)-(1/(2*n)).*cot((ksi2+psi2+pi)./(2*n))...
+(1/(2*n)).*cot((ksi2+phi1+pi)./(2*n))-0.5*cot((ksi2+phi1+pi)./2)-(1/(2*n)).*cot((ksi2+phi2-pi)./(2*n))...
-0.5*cot((ksi2+phi2-pi)./2)+(1/(2*n)).*cot((ksi2+phi2+pi)./(2*n)));
fun2 = exp(-kr*s2.*s2).*F2./cos(ksi2./2); y2(M) = trapz(s2,fun2); error2 = abs(y2(M)-y2_old);
y2_old = y2(M); if (error2<eps1) R2 = y2(M); break; else continue; end; end; I2 = C1*R2;
result = I1+I2+Int_m; % fringe wave given in (4.19)
else
for M=3:Mmax
s = -M:0.0001:M; ksi = -1i*sign(s).*log(1+1i*s.*s+1i*abs(s).*sqrt(s.*s-2*1i));
Fx = ((1/(2*n)).*cot((ksi+psi1-pi)./(2*n))-0.5*cot((ksi+psi1-pi)./2)-(1/(2*n)).*cot((ksi+psi1+pi)./(2*n))...
+(1/(2*n)).*cot((ksi+psi2-pi)./(2*n))-0.5*cot((ksi+psi2-pi)./2)-(1/(2*n)).*cot((ksi+psi2+pi)./(2*n))...
+(1/(2*n)).*cot((ksi+phi1+pi)./(2*n))-0.5*cot((ksi+phi1+pi)./2)-(1/(2*n)).*cot((ksi+phi2-pi)./(2*n))...
-0.5*cot((ksi+phi2-pi)./2)+(1/(2*n)).*cot((ksi+phi2+pi)./(2*n)));
fun = exp(-kr*s.*s).*Fx./cos(ksi./2); y(M) = trapz(s,fun); error = abs(y(M)-y_old);
y_old = y(M); if (error<eps1) Rx = y(M); break; else continue; end; end;
result = C1*Rx; % fringe wave given in (4.19)
end; end; end;
end
