function [design, stimuli, stimuliF] = fDesignStimuli(design, regr)
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
nTargets = design.targetPerBlock; %how many sets of repeating stimuli
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
            
            [freqPattern, transition] = fGenSeq(nTones,design.seqLength,rnd,freqSet,trns,ctr,trg, transTone);
            
            % save sequence and info
            stimuli(c).condLabel = design.conds{c};
            stimuli(c).rnd = rnd;
            stimuli(c).stimulus(s).sequence = freqPattern;
            stimuli(c).stimulus(s).setID = s;
            stimuli(c).stimulus(s).transit = transition;
            
            if s == max(nSets) % cut down the number of trials for the entire experiment
                stimuli(c).stimulus = stimuli(c).stimulus(1:nSets(ci));
                %    stimuli(c).order = randperm(nSets(ci));
                stimuli(c).order = 1:nSets(ci);
                
            end
        end
    end
end

%% Create REGr
if any(strcmp({stimuli.condLabel},'RANREGr'))
    disp('target repeating');
    idx     = find(strcmp({stimuli.condLabel}, 'RANREGr')==1);
    for t = 1:nTargets
        regr(t).sequence = repmat(regr(t).sequence, 1, ceil(design.seqLength/nTones)); %make it long as max seq length, then cut if needed
    end
    regtargets = repmat(regr,1,design.nBlocks*(design.targetRepPerBlock));  %repeat target sequences up till the total number of TARGET stimuli of the experiment
    for st =1:(nTargets*design.nBlocks*(design.targetRepPerBlock)) %attach the N reg targets to the the different rands
        trans = stimuli(idx).stimulus(st).transit;
        list1=stimuli(idx).stimulus(st).sequence(1:trans);
        list2 = regtargets(st).sequence;
        list = [list1 list2(1:(design.seqLength-trans))];
        stimuli(idx).stimulus(st).sequence = list;  %replace targets in stimulus structure
        stimuli(idx).stimulus(st).setID = regtargets(st).setID;  %assign ID of the REG target sequence
    end
end

%% Shuffle order of REGr
if any(strcmp({stimuli.condLabel},'RANREGr'))
    disp('target ordering');
    idx     = find(strcmp({stimuli.condLabel}, 'RANREGr')==1);
    stimuli(idx).order = [];
    st = 1:(nTargets*design.nBlocks*(design.targetRepPerBlock));  %create vector with number of target trials through the experiment
    ord = reshape(st,nTargets,design.nBlocks*(design.targetRepPerBlock)); %subset stimuli per block
    
    for co = 1:(design.targetRepPerBlock):design.nBlocks*(design. targetRepPerBlock) %blocks of repetition
        ordidx = randperm(nTargets); % randomize indx within block
        for r = 1:design.targetRepPerBlock-1
            ordi(:,co) = ord(ordidx, co); % use indices to obtain shuffled matrix
            ordi(:,co+r) = ord(ordidx, co+r); % use indices to obtain shuffled matrix
        end
    end
    ordvector = reshape(ordi, 1, (nTargets*design.nBlocks*(design.targetRepPerBlock))); % reshape matrix in order vector
    stimuli(idx).order = ordvector;
end


%% Create stimuli for familiarity block (take the parameters in StartExp)
if design.familiarityBlock == 1
    nStim = design.FnBlocks * design.FstimPerBlock;
    sizes = unique(design.FcondsSize); % how many different cycle sizes
    stimuliF = struct;
    for i = 1:length(sizes) %loop for 20 and 2 freqPool
        nTones = sizes(i);
        condThisSize = [];
        condsThisSize = find(design.FcondsSize == nTones); % indices of conditions of same size in design.cond
        nSets = design.FnEachCond(condsThisSize);
        
        for s=1:max(nSets)  % make one of each condition per loop, with matched freq content
            freqPool = design.freqPool; % master frequency pool for whole expt
            nTot = length(freqPool);
            pool = randperm(nTot,nTones); % select nTones at random
            freqSet = freqPool(pool);% select out the actual frequencies this corresponds to
            
            %% Generate sequence for a single trial
            for ci = 1:length(condsThisSize)
                c = condsThisSize(ci);
                rnd = design.FcondsRand(c); % logical for REG (0) or RAND(1)
                trns = design.FcondsTrans(c); % logical for get a transition REG-RAND (1) or RAND-REG (1)
                ctr = design.FcondsContr(c); %logical for STEP control condition
                trg = design.FcondsTarget(c); %logical for TARGET condition
                transTone = randi(round(design.transitionTime/design.toneDur)); % transition at which tone, variable interval
                
                [freqPattern, transition] = fGenSeq(nTones,design.seqLengthFAM,rnd,freqSet,trns,ctr,trg, transTone);
                
                % save sequence and info
                stimuliF(c).condLabel = design.Fconds{c};
                stimuliF(c).rnd = rnd;
                stimuliF(c).stimulus(s).sequence = freqPattern;
                stimuliF(c).stimulus(s).setID = s;
                stimuliF(c).stimulus(s).transit = transition; 
                
                if s == max(nSets) % cut down the number of trials for the entire experiment
                    stimuliF(c).stimulus = stimuliF(c).stimulus(1:nSets(ci));
                    stimuliF(c).order = randperm(nSets(ci));
                end
            end
        end
    end
    
    
    regrF = regr;
    for r = 1:nTargets
        regrF(r).sequence =regr(r).sequence(1:design.seqLengthFAM);
    end
    Fregtargets = repmat(regrF,1,design.FnBlocks);
    stimuliF(idx).stimulus = Fregtargets;
    stimuliF(idx).order = 1:(nTargets*design.FnBlocks);
else
    regr =[];
    stimuliF = [];
end
