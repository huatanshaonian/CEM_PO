% ---------------------------------------------------------------------------------------
% Name : sigma12.m
% Author : Feray Hacivelioglu, Levent Sevgi
% Purpose : to calculate the functions (7.76) and (7.77)
% ---------------------------------------------------------------------------------------
function result = sigma12(beta,gamma)
if (gamma >= 0) && (gamma <= pi/2)
    betak = 2*gamma;
elseif (gamma > pi/2) && (gamma <= pi)
    betak = 2*(pi-gamma);
end

if (beta >= 0) && (beta <= betak)
    result = pi-acos((cos(beta)-cos(gamma).^2)./(sin(gamma).^2));
elseif (beta > betak) && (beta <= pi)
    result = 1i*log((cos(gamma).^2-cos(beta)+sqrt((cos(gamma).^2-cos(beta)).^2-sin(gamma).^4))./(sin(gamma).^2))-1i*2*log(sin(gamma));
end
end
