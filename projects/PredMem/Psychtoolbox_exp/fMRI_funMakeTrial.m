   function [trial] = fMRI_funMakeTrial(design,trial,stimuli)
condShuf=design.condShuf{trial.currBlock};
stimShuf=design.stimShuf{trial.currBlock};
ctype = condShuf(trial.trialNum); % numeric code for condition

% fprintf('\nthis is Cond %i.\n', ctype);
% Find which condition it is and extract stimulus properties
trial.condLabel = design.conds{ctype}; 
trial.size = design.condsSize(ctype);  %# Get the numeric characters
trial.rnd = design.condsRand(ctype);
trial.ctr = design.condsContr(ctype);

trial.condLabel = design.conds{ctype};
trial.freqpattern = stimuli(ctype).stimulus(stimShuf(trial.trialNum)).sequence;
trial.condi = ctype;
trial.setID = stimuli(ctype).stimulus(stimShuf(trial.trialNum)).setID;
trial.transit = stimuli(ctype).stimulus(stimShuf(trial.trialNum)).transit;


trial.respond = design.condsTrans(ctype) ; % TODO if you want subject to press button on this trial, e.g. it has a transition or deviant
% fprintf('\nthis is ID %i.\n', trial.setID);
% disp(trial.trialNum);

% Now generate list of tones
trial.stim = [];

for l=1:numel(trial.freqpattern)
    tone = fgenTone_adjust(trial.freqpattern(l), design.toneDur, design.fs);
    trial.stim = [trial.stim tone];
end

trial.stim=trial.stim/max(abs(trial.stim));
trial.stim=trial.stim*0.8;

% End condition
if trial.trialNum == (design.stimPerBlock * design.nBlocks)
    trial.done = 1;
end

end
