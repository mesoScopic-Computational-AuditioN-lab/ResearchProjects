function all_freqs = generate_frequencies_pupil(cfg, segmentz)

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
all_freqs       = arrayfun(@(x) nan(1, x), segmentz(14, :), 'UniformOutput', false);           % all frequencies that are presented (return value)

% loop over blocks % runs
for b = 1: size(all_freqs, 2)
    
    stim_freqs      = nan(2, segmentz( 14, b));         % stimuli for this block
    seglen          = segmentz(14, b);                  % length of segment

    % loop over center frequencies
    for i = 0:cfg.n_freq-1
        freq    = segmentz(9+i, b);                     % get freq A and B for this block
        prob    = segmentz(6+i, b);                     % get prob A and B for this block
        octv    = segmentz(8, b);                       % get octv width coupled to frequency

        stim_freqs(i+1, :)    = 2.^normrnd(freq, octv, [1, seglen]);         % populate [centerfreq x tpb] with samples
    end
    
    boolprob     =  2 - (rand(seglen, 1) < segmentz(6, b));      % from what centerfreq to sample
    lin_idx      =  boolprob.' + (0:size(stim_freqs, 2)-1)*2;    % get correct matrix indixing - add row multiplier
    all_freqs{b} =  stim_freqs(lin_idx);                         % use indexing to get frequencies and save per block 

end

% if disired convert to some octive resolution
if isnumeric(cfg.oct_res)
    for b = 1: size(all_freqs, 2)
        all_freqs{b}       = 2.^(round(log2(all_freqs{b}) * cfg.oct_res)/cfg.oct_res);               % round using octive resolution
    end
end
return