function all_freqs = generate_frequencies_main(cfg, segmentz)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                  %
%   GENERATE FREQUENCIES FOR MAIN EXPERIMENT                       %
%     Input:                                                       %
%       - cfg struct: settings of main exp                         %
%       - segments: blocking/counterbalancing                      %
%     Returns:                                                     %
%       - all_freqs: matrix of [blocks x presented frequency       %
%       - segments: blocking/counterbalancing                      %
%                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SAMPLE AROUND CENTER FREQUENCIES
stim_freqs      = nan(cfg.n_freq, cfg.tpb);         % stimulus frequencies for current block
all_freqs       = nan(cfg.n_blocks, cfg.tpb);       % all frequencies that are presented (return value)
boolprob        = nan(1, cfg.tpb);                  % a boolean array to sellect for where to sample

% loop over blocks
for b = 1: cfg.n_blocks

    % loop over center frequencies
    for i = 1:cfg.n_freq
        freq                = segmentz(7+i,(segmentz(2,:)==b)&(segmentz(3,:)==1));     % get freq A and B for this block
        octv                = cfg.oct_width(cfg.cent_freqs == freq);                   % get octv coupled to frequency
        stim_freqs(i, :)    = 2.^normrnd(freq, octv, [1, cfg.tpb]);                    % populate [centerfreq x tpb] with samples
    end

    % loop over segments and get probabilities
    for s = 1:cfg.n_segments
        tr_segment = [((s-1)*cfg.tps)+1, s*cfg.tps];                                   % get indexes of trials relative to block
        boolprob(tr_segment(1): tr_segment(2)) = 2 - (rand(cfg.tps,1) ...              % from what to sample (1==cf_A, 2==cf_B)
                                                < segmentz(6,segmentz(2,:) == b ...
                                                & segmentz(3,:) == s));  
    end

    % finally take boolean index and pick correct sample
    lin_idx                 = boolprob+(0:size(stim_freqs,1):numel(stim_freqs) ...     % get correct matrix indixing - add row multiplier
                                        -size(stim_freqs,1));  
    all_freqs(b, :)         = stim_freqs(lin_idx);                                     % use indexing to get frequencies and save per block 
end

% if disired convert to some octive resolution
if isnumeric(cfg.oct_res)
    all_freqs       = 2.^(round(log2(all_freqs) * cfg.oct_res)/cfg.oct_res);               % round using octive resolution
end
return