%% Optimization (Minimizaiton) Solver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Find the Minimizer of a Function    
%
%   Args:
%       funopts: cellarray for objective function options {beta, budget, grid, Ngrid, expvf}
%       lb: lower bound of minimizer
%       ub: upper bound of minimizer
%
%   Returns:
%       aprime: (scalar) minimizer of objective function
%
function [aprime] = fnOptaprime(funopts, lb, ub)
    [beta, budget, grid, Ngrid, expvf] = deal(funopts{:});
    options = optimset('Display', 'off', 'MaxIter', 2000, 'tolX', 1e-10);
    [aprime, ~] = fminbnd((@(x) fnHHobj(x, beta, budget, grid, Ngrid, expvf)), lb, ub, options);
end