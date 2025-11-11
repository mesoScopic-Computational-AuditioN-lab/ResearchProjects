function x_array = logistic_func(n_probs, K)
%% Create probability array using the logistic function

% predefine matrix for probabilities
x_array = nan(2, n_probs);

% loop over x's and calculate using logistic function the probability
for x = 1:n_probs
    x_array(1, x) = 1 / (1 + exp(-K * (x - ((n_probs+1)*0.5))));
end
x_array(2, :) = 1-x_array(1, :);    % also fill in the counter probability


return;