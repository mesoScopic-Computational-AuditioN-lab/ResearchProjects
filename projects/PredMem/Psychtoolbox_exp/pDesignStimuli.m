function [design, stimuli] = pDesignStimuli(design)
%% function to generate all tone-pip sequences for entire RR experiment
% Stimuli are designed at the start of the experiment, and
% the audio is generated at each trial.
%
% Pre-generating the sequences allows matching of the specific tones used
% between instances of different conditions with the same cycle size -
% referred to here as a 'set'.
% i.e. one set for size 5 would use the same 5 tones and same additional
% deviant tone to generate the RAND, REG, RANDdev and REGdev variants.
%
%
% various options are available for constraining the choices of
% frequencies. These are set in the input structure design; which has been
% initialised in Start_Experiment or GenerateExamples
%
% -- design.drawReplace
%   choose sequence frequencies
%       0: without replacement
%       1: with replacement
% -- design.noRep
%       0: allow adjacent tones to repeat in RAND
%       1: do not allow (re-shuffle until condition satisfied)
%
%______________________________Rosy Southwell, last updated Feb 27th 2017
%______________________________Roberta Bianco, last updated Oct 2017

%% Check inputs & use defaults for missing fields in design structure
nStim = design.nBlocks * design.stimPerBlock;
sizes = unique(design.condsSize); % how many different cycle sizes
%nSets = nStim/length(design.conds); % how many sets of stimuli needed generating together
nTargets = design.targetPerBlock; %how many sets of target stimuli
stimuli = struct;
% rng('shuffle');
rng(str2num(design.subnum))

for i = 1:length(sizes) %loop for 20 and 2 freqPool
    nTones = sizes(i);
    condsThisSize = find(design.condsSize == nTones); % indices of conditions of same size in design.cond
    nSets = design.nEachCond(condsThisSize);
    
    for s=1:max(nSets)  % make one trial of each condition with same alphabet size per loop, with matched freq content; make many trials
        freqPool = design.freqPool; % master frequency pool for whole expt
        nTot = length(freqPool);
        pool = randperm(nTot,nTones); % select nTones at random
        freqSet = freqPool(pool);% select out the actual frequencies this corresponds to
        
        %% Generate sequence for a single trial
        for ci = 1:length(condsThisSize) %loop through each condition with same alphabet size
            c = condsThisSize(ci);
            rnd = design.condsRand(c); % logical for REG (0) or RAND(1)
            trns = design.condsTrans(c); % logical for get a transition REG-RAND (1) or RAND-REG (1)
            ctr = design.condsContr(c); %logical for STEP control condition
            trg = design.condsTarget(c); %logical for TARGET condition
            transTone = randi(round(design.transitionTime/design.toneDur)); % transition at which tone, variable interval
            
            if numel(design.noRep) == 1
                noRep = design.noRep;
            else
                noRep = design.noRep(c);
            end
            [freqPattern, transition] = fGenSeq(nTones,design.seqLength,rnd,freqSet,trns, ctr, trg, transTone); %from freqSet:noRep,trns,ctr,trg);
            
            % save sequence and info
            stimuli(c).condLabel = design.conds{c};
            stimuli(c).rnd = rnd;
            stimuli(c).stimulus(s).sequence = freqPattern;
            stimuli(c).stimulus(s).setID = s;
            stimuli(c).stimulus(s).transit = transition;
            stimuli(c).stimulus(s).rew = 0;
            
            if s == max(nSets) % cut down the number of trials for the entire experiment
                stimuli(c).stimulus = stimuli(c).stimulus(1:nSets(ci));
                stimuli(c).order = randperm(nSets(ci));
            end
        end
    end
end
