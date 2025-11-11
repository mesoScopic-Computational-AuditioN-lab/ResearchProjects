function K = calc_logistic_growth(f0, array_len)
% calculate the growth rate given the length op an array and the 0th point
% percentage (e.g. 0.1 = 10%)

syms x;
eqn = 1 / (1 + exp(-x*((array_len-1) * 0.5))) == f0;
K = solve(eqn);

return;