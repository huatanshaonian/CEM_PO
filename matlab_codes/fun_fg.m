% ---------------------------------------------------------------------------------------
% Name : fun_fg.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate f1 and g1 directivity patterns of (4.20) and (4.21)
% ---------------------------------------------------------------------------------------
function [f1,g1] = fun_fg(angle,angle0,alfa)
eps = 1e-5; % 0<eps<<1 in radians
psi1 = angle-angle0; psi2 = angle+angle0; n = alfa/pi;
angle1 = (pi-psi2)/(2*n); angle2 = (pi-psi1)/(2*n);
angle3 = (pi+psi2)/(2*n); angle4 = (pi+psi1)/(2*n);

if (angle0>=eps) && (angle0<=alfa-pi-eps) % in case of single-side illumination
    if (psi1 <= pi+eps) && (psi1 >= pi-eps)
        f1 = (1/(2*n))*(cot(angle1)+cot(angle3)-cot(angle4))-0.5*cot((pi-psi2)/2);
        g1 = -(1/(2*n))*(cot(angle1)+cot(angle3)+cot(angle4))+0.5*cot((pi-psi2)/2);
    elseif (psi2 <= pi+eps) && (psi2 >= pi-eps)
        f1 = (1/(2*n))*(-cot(angle2)+cot(angle3)-cot(angle4))+0.5*cot((pi-psi1)/2);
        g1 = -(1/(2*n))*(cot(angle2)+cot(angle3)+cot(angle4))+0.5*cot((pi-psi1)/2);
    else
        f = (1/(2*n))*(cot(angle1)-cot(angle2)+cot(angle3)-cot(angle4));
        g = -(1/(2*n))*(cot(angle1)+cot(angle2)+cot(angle3)+cot(angle4));
        f0 = 0.5*(cot((pi-psi2)/2)-cot((pi-psi1)/2));
        g0 = -0.5*(cot((pi-psi2)/2)+cot((pi-psi1)/2));
        f1 = f-f0; g1 = g-g0;
    end
elseif (angle0>=alfa-pi+eps) && (angle0<=pi-eps) % case of double-side illumination
    if (psi2 <= pi+eps) && (psi2 >= pi-eps)
        f1 = (1/(2*n))*(cot(angle3)-cot(angle4))-0.5*cot((pi-psi2)/2);
        g1 = -(1/(2*n))*(cot(angle3)+cot(angle4))+0.5*cot((pi-psi2)/2);
    elseif (2*alfa-psi2 <= pi+eps) && (2*alfa-psi2 >= pi-eps)
        f1 = (1/(2*n))*(cot(angle1)-cot(angle2))+0.5*cot((pi-2*alfa+psi2)/2);
        g1 = -(1/(2*n))*(cot(angle1)+cot(angle2))-0.5*cot((pi-2*alfa+psi2)/2);
    else
        f = (1/(2*n))*(cot(angle1)-cot(angle2)+cot(angle3)-cot(angle4));
        g = -(1/(2*n))*(cot(angle1)+cot(angle2)+cot(angle3)+cot(angle4));
        f0 = 0.5*(cot((pi-psi2)/2)+cot((pi-2*alfa+psi2)/2));
        g0 = -0.5*(cot((pi-psi2)/2)+cot((pi-2*alfa+psi2)/2));
        f1 = f-f0; g1 = g-g0;
    end
    elseif (angle0 == angle) && (angle == pi/2)
        f1 = (1/(2*n))*(-cot(angle2)+cot(angle3)-cot(angle4));
        g1 = -(1/(2*n))*(cot(angle2)+cot(angle3)+cot(angle4));
end
