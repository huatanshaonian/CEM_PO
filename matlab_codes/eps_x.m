function y = eps_x(x)
% Helper function for Eq (7.48)
% eps(x) = 1 if 0 < x <= pi
%          0 if pi < x
if (x > 0) && (x <= pi)
    y = 1;
else
    y = 0;
end
end
