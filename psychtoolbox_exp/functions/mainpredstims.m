function mainpredstims(ppnum, wrun, setup)

settings_main;

%% CONDITIONS AND COUNTERBALANCING

nsegments           = cfg.n_segments;                                       % number of segments per block
nblocks             = cfg.n_blocks;                                         % number of blocks

nadpts              = 3;                                                    % number of adaptation cohorts
npreds              = 3;                                                    % number of prediction cohorts

nprobs              = cfg.n_probs;                                          % number of probabilities
npairs              = size(cfg.block_pairs, 2);                             % number of freq pairs

nsegments_all       = nsegments * nblocks;                                  % total number of segments                 

% block counterbalance
cb_blocks           = counterbalance([npairs], nblocks/npairs, [], 1, [], 'subdiv');
cb_segments         = counterbalance([nprobs], 1, [], nsegments_all/nprobs, [], 'full');

segmentz = NaN(8, nblocks*nsegments);
segmentz( 1,  :)     = repelem(1:nblocks, nsegments);                       % what block
segmentz( 2,  :)     = repmat(1:nsegments, 1, nblocks);                     % trial in block
segmentz( 3,  :)     = cb_segments;                                         % prob condition per segment
segmentz( 4,  :)     = repelem(cb_blocks, nsegments);                       % what freq pair is being presented

segmentz( 5,  :)     = cfg.stim_probs(1, segmentz(3, :));                   % probability of selecting A in section
segmentz( 6,  :)     = cfg.stim_probs(2, segmentz(3, :));                   % probability of selecting B in section

segmentz( 7,  :)     = cfg.cent_freqs(cfg.block_pairs(1, segmentz(4,:)));   % center freq of A this block
segmentz( 8,  :)     = cfg.cent_freqs(cfg.block_pairs(2, segmentz(4,:)));   % center freq of B this block

% aproximate predictablity and adaptability of blocks
pred_idx = nan(1,nblocks);
adpt_idx = nan(1,nblocks);
for b = 1:nblocks
    pred_idx(b) = std(diff(cfg.stim_probs(1, segmentz(3, segmentz(1,:) == b))));
    adpt_idx(b) = mean(cfg.stim_probs(1, segmentz(3, segmentz(1,:) == b)));
end
[~, pred_idx] = sort(pred_idx);
[~, adpt_idx] = sort(adpt_idx);

% set segments predictability
for i = 1:nadpts
    segmentz(9, ismember(segmentz(1, :), pred_idx(repelem(1:nadpts, nblocks/nadpts) == i)))     = i;    % pedictability (1=low-switching, 2=medium-swithing, 3=high-switching)
    segmentz(10, ismember(segmentz(1, :), adpt_idx(repelem(1:npreds, nblocks/npreds) == i)))    = i;    % adaptability (1=A-higher, 2=average, 3=B-higher)
end


%% GENERATE TONES

% generate tones (or load if they were already created)
disptext(w, 'Generating now...(please wait)', screenrect, ...
            cfg.visual.fontsize, cfg.visual.textfont, cfg.visual.textcol);
Screen('Flip', w);
% actually generate tones
pres_freq               = generate_frequencies_main(cfg, segmentz);
[~, mod_tones]          = create_tones_main(pres_freq, cfg.stim_t, cfg.sound.samp_rate, ...
                                                cfg.ramp_ops, cfg.ramp_dns);
save (fullfile( pwd, 'stimuli', ppnum, [ppnum '-r' wrun '_main_stims.mat']), 'mod_tones', 'pres_freq', 'segmentz');

end