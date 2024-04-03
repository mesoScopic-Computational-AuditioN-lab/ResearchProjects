function rundrex_stims(pp, input_dir)
% Usage: display_DREX_output(mdl, x)
% Wrapper function to take path and pp naming,
%   load stimuli, run drex model per block,
%   and save .mat file.
% Also saves the input oct_range and stimuli as senity checlk
%
% Optional parameters for .mat naming convention and
%   drex parameters are specified WITHIN the function.
%
% Input Arguments:
%   pp              current participant number (string)
%   input_dir       parrent directory (string)

%% Settings

% input data settings
fn_suffix   = '_stimdf.mat';      % suffix of input filename
outfn_suffix = '_drexdf.mat';     % suffix of output filename
use_oct     = true;               % if false use frequencie space instead

% drex parameters
distr       = 'gmm';       % drex distribution to use, gmm=gaussian mixture 
n_comp      = 2;           % maximum number of componenent (gaussians)
beta        = 0.2;         % beta to use
D           = 1;           % D
priors      = 'minmax';    % if minmax, use minimum and maximum of that blocks range as prior else {[priors]}
maxhyp      = inf;         % limits number of context hypotheses, pruning the beliefs
memory      = inf;         % memory length to use


% load the actual stim df data, must contain stims, freq_range, & oct_range
load( fullfile( input_dir, pp, [pp fn_suffix]), 'freq_range', 'oct_range', 'stims');

if use_oct
    s_range = oct_range;
    stimuli = stims.frequencies_oct;
else
    s_range = freq_range;
    stimuli = stims.frequencies;
end

% predefine length of output arrays
prob_array = NaN(length(s_range), length(stimuli));
surp_array = NaN(1, length(stimuli));

% loop over blocks and get probabilities and surprise
for blk = unique(stims.block)
    
    % take stumuli of one block
    x = stimuli(stims.block == blk)';
    
    % set parameters
    params = [];
    params.distribution = distr;
    params.max_ncomp = n_comp;
    params.beta = beta;
    params.D = D;
    params.prior = estimate_suffstat(x,params);
    if strcmp(priors,'minmax')
        params.prior.mu = {[min(x) max(x)]};
    else
        params.prior.mu = {priors};
    end

    params.maxhyp = maxhyp;
    params.memory = memory;
    
    % run actual drex model given parameters
    out = run_DREX_model(x,params);
    
    % then calculate in our frequency space the likelihood
    [PD,X,Y] = post_DREX_prediction(1,out,s_range);

    % populate surprisal and prob
    surp_array(1, (blk-1)*length(x)+1:blk*length(x)) = out.surprisal;
    prob_array(:,(blk-1)*length(x)+1:blk*length(x)) = PD;
end

%% sanity check plotting functions
%contourf(prob_array,'fill','on','linestyle','-','linecolor','none')
%plot(surp_array)

%% Save .mat file for later use
save( fullfile( input_dir, pp, [pp outfn_suffix]), 'surp_array', 'prob_array', 's_range', 'stimuli');

end