function[list, transTone] = fGenSeq(nTones,seqLength,rnd,freqSet,trns, ctr, trg, transTone)
%% This function generates a vector representing frequency sequence in a single trial
%
% INPUTS
% -- design: a structure of parameters describing the entire experiment

% parameters for THIS trial:
% -- nTones: how many unique frequencies in the sequence
% -- seqLength: how many tone pips in a stimulus
% -- rnd
%       0: sequences are those which follow a cyclical pattern.
%       1: sequences are made to match each REG sequence, but with a shuffled
%           order
% -- freqSet: the exact frequencies to use in the sequence this trial
% -- noRep: determines the shuffle type for RAND
%       0: allow adjacent repeats
%       1: re-shuffle until no adjacent repeats
% -- trns: determines if there is a transition to REG or RAND
%       0: no transition, stay REG or RAND depending on rnd
%       1: transition, from REG or RAND depending on rnd
% -- ctr: determines the STEP condition, transition form one tone to another
% -- trg: determines the RANDREG that are TARGET condition
%       0: RAND and REG part change always
%       1: REG part is repeated N times through blocks
% OUTPUTS
% -- list: list of frequencies for the sequence
% -- transtone: tone where there is a transition
%
%______________________________Roberta Bianco, last updated Oct 2017


%% Generate sequence REG and RAND by default
listreg = [];
listshf = [];
freqSetShuf = freqSet(randperm(length(freqSet)));% shuffle frequency subset));

if ctr == 0  %for non control conditions, generate REG sequence by default with frequency subset
    %% Generate REG sequence from Poolset
    freqSetShuf = freqSet(randperm(length(freqSet)));
    % generate regular sequence
    listreg = repmat(freqSetShuf,1,ceil(seqLength/nTones)); % repeat the regular sequence
    % cut the regular sequence off at the sequence length (in case
    % seq_length/ntones isn't an integer)
    listreg = listreg(1:seqLength);
    
    %% Generate RAN sequence from REG
    listshf = listreg(randperm(seqLength));
    % check to make sure there aren't any repeated tones
    while sum(diff(listshf)==0)>0
        % if there are repeated tones, swap one of those tones with another
        % tone
        rep_idx = find(diff(listshf)==0)+1;
        for ii = 1:length(rep_idx) % do this for each repeated tone
            % get possible places to swap to
            poss_swap_spots = setxor(1:seqLength,[rep_idx-1 rep_idx]);
            % randomly pick a spot
            swap_to_this_spot = poss_swap_spots(randi(length(poss_swap_spots)));
            % do the swap
            listshf([swap_to_this_spot rep_idx(ii)]) = listshf([rep_idx(ii) swap_to_this_spot]);
        end
    end
    
    %% create the sequences according to constraints from the listshf (created from full or subset pool) and listreg (made from subset of pool)
    if rnd
        if trns  %generate RANDREG if transition
            list1=listshf(1:transTone-1);
            list2 = listreg;
            if trg  % preset ran and reg sequences with full sequence length (rand and reg are matched if required, and frequency pool is matched with other conditions)
                list = [listshf];
            else  % add cut REG sequence to RAND
                list = [list1 list2(1:(seqLength-transTone+1))];
            end
        else
            list=listshf;  % generate only RANDs if not trns
            transTone = 0;
        end
        
    else
        if trns %generate REGRAND if transition
            list1 =listreg(1:transTone-1);
            list2 = listshf;
            list = [list1 list2(1:(seqLength-transTone+1))];
        else
            list = listreg; %  %generate REG sequence if not trns
            transTone = 0;
        end
    end
    
else %control condition STEP
    if trns %generate step if transition
        list = repmat(freqSetShuf(1),1,seqLength);
        list1 = list(1:transTone-1);
        list2 = repmat(freqSetShuf(2),1,seqLength);
        list = [list1 list2(1:(seqLength-transTone+1))];
    else
        list = repmat(freqSetShuf(1),1,seqLength); %  continuous tone repetition
        transTone = 0;
    end
end

end %end function