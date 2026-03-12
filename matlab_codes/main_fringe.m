% ---------------------------------------------------------------------------------------
% Name : main_fringe.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate and compare normalized exact and asymptotic fringe fields
% ---------------------------------------------------------------------------------------
% ***************************************************************************************
% ******************************************* Input parameters **************************
% ***************************************************************************************
Fringe = 1; Fringe_Asym = 1;
alfamax = 300; % wedge angle [Deg]
angle0 = 45; % incident angle [Deg]
% ***************************************************************************************
alfa = alfamax*pi/180; % change wedge angle degree to radians
angle0 = angle0*pi/180; % change incident angle degree to radians
x = 1; % r = x*lamda; r:observation distance
kr = 2*x*pi;
A = 0:2.0:alfamax; % observation angle
[dd nA]=size(A); Ar=A*pi/180; % change observation angle degree to radians
% ***************************************************************************************
% --------------------------------------- Exact Fringe Fields ---------------------------
if (Fringe == 1)
    for m = 1:nA
        fprintf(1, 'Calculating Exact Fringe Field = > Angle %5.3f : \n',A(m));
        angle = Ar(m);
        % Absolute values of the normalized exact fringe fields: | u(1)/u0 |
        Us_Fringe(m) = abs(Int_calcFringe(alfa,kr,angle0,angle,'Soft')); % for soft boundary conditions
        Uh_Fringe(m) = abs(Int_calcFringe(alfa,kr,angle0,angle,'Hard')); % for hard boundary conditions
    end
end
% ------------------------------------ Asymptotic Fringe Fields -------------------------
if (Fringe_Asym == 1)
    coeff = exp(1i*(kr+pi/4))/sqrt(2*pi*kr);
    for m = 1:nA
        fprintf(1, 'Calculating PWA Fringe Field = > Angle %5.3f : \n',A(m));
        angle = Ar(m);
        [f1,g1] = fun_fg(angle,angle0,alfa); % directivity patterns
        % Absolute values of the normalized asymptotic fringe fields : | u(1)/u0 |
        Us_Fringe_PWA(m) = abs(f1.*coeff); % for soft boundary conditions
        Uh_Fringe_PWA(m) = abs(g1.*coeff); % for hard boundary conditions
    end
end
% -------------------------------------------- END --------------------------------------
% ******************************************* FIGURES ***********************************
figure(1)
polar(Ar,Us_Fringe,'k'); hold on;
polar(Ar,Us_Fringe_PWA,'k:'); hold on;
legend('Exact','Asymp');
title('Soft Boundary Condition');

figure(2)
polar(Ar,Uh_Fringe,'k'); hold on;
polar(Ar,Uh_Fringe_PWA,'k:'); hold on;
legend('Exact','Asymp');
title('Hard Boundary Condition');
