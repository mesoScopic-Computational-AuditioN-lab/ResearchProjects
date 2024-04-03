function [trial, res] = fnewTrial_fam(res,design,trial,stimuliF)
condShuf=design.FcondShuf{trial.currBlock};
stimShuf=design.FstimShuf{trial.currBlock};
ctype = condShuf(trial.trialNum); % numeric code for condition
fprintf('\nthis is Cond %i.\n', ctype);

% Find which condition it is and extract stimulus properties
trial.condLabel = design.Fconds{ctype};
trial.size = design.FcondsSize(ctype);  %# Get the numeric characters
trial.rnd = design.FcondsRand(ctype);
trial.ctr = design.condsContr(ctype);

trial.condLabel = design.Fconds{ctype};
trial.freqpattern = stimuliF(ctype).stimulus(stimShuf(trial.trialNum)).sequence;
trial.condi = ctype;
trial.setID = stimuliF(ctype).stimulus(stimShuf(trial.trialNum)).setID;
%trial.transit = stimuliF(ctype).stimulus(stimShuf(trial.trialNum)).transit;
trial.respond = design.FcondsTarget(ctype) ; % if you want subject to press button on this trial, e.g. it has a familiar pattern
fprintf('\nthis is ID %i.\n', trial.setID);

% Put lists of stimulus frequencies in results structure
res.Ffreqlist{trial.trialNum} = trial.freqpattern;
res.Fcondi(trial.trialNum) = ctype;

% Now generate list of tones
trial.stim = [];

    for l=1:numel(trial.freqpattern)
        tone = fgenTone_adjust(trial.freqpattern(l), design.toneDur, design.fs);
        trial.stim = [trial.stim tone];
    end
trial.stim=trial.stim/max(abs(trial.stim)); %%% ADD NORMALIZATION
trial.stim=trial.stim*0.8;
trial.stim = [trial.stim zeros(1,design.fs/4)]; % add 250 ms at end of stim

% End condition
if trial.trialNum == (design.stimPerBlock * design.nBlocks)
    trial.done = 1;
end

end
