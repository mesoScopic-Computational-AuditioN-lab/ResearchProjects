function fMRI_funSaveWav(design,stimuli, set)
% Save long wav files with mini trials in randomised order
% Roberta Bianco 2023

for b = 1:design.nBlocks
    trial.currBlock = b;

    % Filename and path
    outfile = sprintf('sub%s_run%d_set%d', design.subnum, b, set);
    % First, randomise MATLAB's random number generator
    % rand('seed',sum(100*clock)); % for older versions of MATLAB
    rng('shuffle');   % completely random
    % rng('default')

%% RANDOMISE STIMULI ORDER
    % Generate stimulus presentation order (should be randomised)
    condIDs=[];     stimIDs=[];
    for c= 1:numel(design.conds) %for all main conditions
        % Each condition is assigned a unique integer i between 1 and Nconds.
        % this corresponds to the order they are specified in Start_experiment
        % The reason why we want to balance stimulus numbers within each block
        % is because it makes it possible to compute d primes per block, and
        % also makes it easier to resume the experiment
        condIDs=[condIDs c*ones(1,design.nEachCond(c)/design.nBlocks)]; %condition IDs (e.g.:1 to 6)
        stimIDs =[stimIDs ...
            stimuli(c).order((trial.currBlock-1)*design.nEachCond(c)/design.nBlocks+1 : ...
            trial.currBlock*design.nEachCond(c)/design.nBlocks)]; % stim IDs, repetition index per each block
    end

    shufOrder = randperm(design.stimPerBlock);
    stimTarget = stimIDs(condIDs==1);  %N targets are repeated N times withing block with the same order
    design.condShuf{b} = condIDs(shufOrder);
    design.stimShuf{b} = stimIDs(shufOrder);
    idx = design.condShuf{b} == 1;   %finds where targets are randomly placed in the shuffled sequence of stimuli
    if any(strcmp({stimuli.condLabel},'RANREGr'))
        design.stimShuf{b}(idx) = stimTarget; %replace with the predefined order
    end

    stimTarget = stimIDs(condIDs==3);  %N targets are repeated N times withing block with the same order
    idx = design.condShuf{b} == 3;   %finds where targets are randomly placed in the shuffled sequence of stimuli
    if any(strcmp({stimuli.condLabel},'RANRANr'))
        design.stimShuf{b}(idx) = stimTarget; %replace with the predefined order
    end
    design.nStimTot    = numel(condIDs)*design.nBlocks;

%% CONCATENATE TRIALS 
    freqlist =[];
    infomat = [];
    for i=design.startTrial:(design.stimPerBlock)
        %% start trial
        trial.trialNum  = i;
        % Gets the next trial; some of the trial parameters are stored in a
        % 'trial' struct
        [trial] = fMRI_funMakeTrial(design,trial,stimuli);
        trial.stim = [trial.stim; trial.stim]; % make it stereo
        onset = length(freqlist)*50;   % hardcoded, make `design.toneDur` ?
        freqlist =  [freqlist trial.freqpattern];
        info = [trial.trialNum trial.condi trial.setID onset];
        infomat = [infomat; info];
    end

%% SAVE WAV FILE 
    stim=[];
    for l=1:numel(freqlist)
        tone = fgenTone_adjust(freqlist(l), design.toneDur, design.fs);
        stim = [stim tone];
    end
    
    stim=stim/max(abs(stim));
    stim = [stim zeros(0.2*design.fs,1)'];
    stim=stim*.8;  %normalizing (to avoid clipping)

    audiowrite([design.resultsDir outfile '.wav'], stim, design.fs);
    save([design.resultsDir outfile '.mat'],'infomat', 'freqlist');

    disp(outfile);
    clear freqlist infomat stim

end

