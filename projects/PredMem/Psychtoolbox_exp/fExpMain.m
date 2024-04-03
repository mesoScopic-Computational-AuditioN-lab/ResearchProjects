function fExpMain(design,stimuli,stimuliF)
% Main experiment function for all Rand Reg experiments. This
% function generates audio on each trial, records responses (if applicable)
% and analyses any behavioural data (minimally).
% If necessary it sends triggers for the EEG
%
%__________________________________________________________________________
% FORMAT fExpMain(design, stimuli)
%
% -- design : structure containing details about the experimental
%                    design, see parameters in Start_EEG_Experiment
% -- stimuli : actual frequency sequences to use
% sca to quite the psychtoolbox window
% q to stop the experiment
%__________________________________________________________________________
% Rosy Southwell 2017/02
% Roberta Bianco 2017/10

clc;
commandwindow; % switch cursor to command window if not already done

%% Preparing a little Matlab matrix with the image of fixation crosses:
FixCr           = ones(20,20)*0;
FixCr(10:11,:)  = 255;
FixCr(:,10:11)  = 255;  %try imagesc(FixCr) to display the result in Matlab

%% Start PTB
%  when working with the PTB it is a good idea to enclose the whole body of
%  your program in a try ... catch ... end construct. This will often prevent
%  you from getting stuck in the PTB full screen mode
try
    
    % Perform PTB and Eyetracker initialisation (see inside for details)
    [pahandle, expWin, rect]=fInitialisePTB(design);
    
    % Get the midpoint (mx, my) of this window, x and y
    [mx, my] = RectCenter(rect);
    Screen('Preference', 'SkipSyncTests', 0); % if problems with psychtoolbox screen sync, set to 1
    
    % This is our intro text.
    myText = ['Are you ready for the experiment?\n' ...
        '(Press <SPACEBAR> to continue)\n' ];
    
    % Draw 'myText', centered in the display window:
    DrawFormattedText(expWin, myText, mx/2, 'center', 255, 0, [], [], 1.1);
    
    % Show the drawn text at next display refresh cycle:
    Screen('Flip', expWin);
    
    % Wait for key stroke. This will first make sure all keys are
    %released, then wait for a keypress and release:
    KbQueueWait;
    
    
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Start experiment
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    trial.done      = 0; % flag=1 when experiment finished
    if design.whichBlock <= design.nBlocks
        for b = design.whichBlock:(design.whichBlock+design.nBlocksRun-1)
            trial.currBlock = b;
            
            % Filename and path
            
            outfile = sprintf('log_sub%s_block%d', design.subnum, ...
                trial.currBlock);
            
            outpath = design.resultsDir;
            
            % First, randomise MATLAB's random number generator
            % rand('seed',sum(100*clock)); % for older versions of MATLAB
            rng('shuffle');   % completely random
            % rng('default')
            
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
            stimTarget = stimIDs(1:design.targetPerBlock*(design.targetRepPerBlock));  %N targets are repeated N times withing block with the same order
            design.condShuf{b} = condIDs(shufOrder);
            design.stimShuf{b} = stimIDs(shufOrder);
            idx = design.condShuf{b} == 1;   %finds where targets are randomly placed in the shuffled sequence of stimuli
            if any(strcmp({stimuli.condLabel},'RANREGr'))
                design.stimShuf{b}(idx) = stimTarget; %replace with the predefined order
            end
            
            disp(design.condShuf{b})
            disp(design.stimShuf{b})
            design.nStimTot    = numel(condIDs)*design.nBlocks;
            
            if ~design.passive
                design.rKey        = KbName('Space'); %response key is Space bar;
            end

            %% Set up a 'results' structure. This contains subjects results and stats
            results.condi       = NaN*ones(1,design.stimPerBlock);   	%to  be filled in after each trial
            results.freqlist    = [];                                   %list of freqs used
            if ~design.passive
                results.response    = NaN*ones(1,design.stimPerBlock);      %to  be filled in after each trial
                results.keypress    = NaN*ones(1,design.stimPerBlock);   	%stores all keypresses
                results.RTs         = NaN*ones(1,design.stimPerBlock);   	%reaction times
                results.correct     = NaN*ones(1,design.stimPerBlock);
            end
            
            %% Set up a 'trial' structure. This contains all info pertaining to current trial
            trial.keypress  = [];
            
            if ~design.passive
                % Only record q and space, and enter keys
                keycodes=zeros(1,256);
                keycodes(KbName({'l','z','q', 'space', 'Return'}))=1;
                KbQueueCreate([], keycodes); % Restrict to certain keys
                KbQueueStart(); % Start recording keypresses
            end
            
            
            % Draw fixation cross and wait for keypress
            myText1 = ['In this task you are asked to detect occasional CHANGES\n' ...
                '(transitions from RANDOM to REGULAR patterns OR from one single tone to another tone)\n'...
                'occurring during a rapid sequence of tones.\n' ...
                ' \n'...
                '>>> Press  <SPACEBAR> as fast as possible if you hear a change \n' ...
                '>>> Do NOT press any button if there is no change \n' ...
                ' \n' ...
                'You will receive feedback throughout the experiment (green circles for fast responses,\n' ...
                'orange for medium, and red for slow ones).\n ' ...
                ' \n' ...
                '(Press SPACEBAR to continue)\n' ];
            fixcross=Screen('MakeTexture',expWin,FixCr);
            DrawFormattedText(expWin, myText1, mx/2, 'center', 255, 0, [], [], 1.1);
            Screen('Flip', expWin);
            KbQueueWait;
            WaitSecs(1);
            for i=design.startTrial:(design.stimPerBlock)
                %% start trial
                trial.trialNum  = i;
                Screen('Flip', expWin);
                KbQueueFlush; %Empty keyboard queue before each trial
                
                Screen('DrawTexture', expWin, fixcross,[],[mx-10,my-10,mx+10,my+10]);
                Screen('Flip', expWin);
                
                % Gets the next trial; some of the trial parameters are stored in a
                % 'trial' struct
                [trial,results] = fnewTrial(results,design,trial,stimuli);
                trial.stim = [trial.stim; trial.stim]; % make it stereo

                % If recording EEG, add a trigger channel to the stimulus matrix
                if design.EEG
                    trial = fTrigger(trial,design);
                end
                
                % Fill the audio playback buffer with the audio data 'wavedata':
                PsychPortAudio('FillBuffer', pahandle, trial.stim);
                
                if ~design.passive
                    % Only record q and space, and enter keys
                    keycodes=zeros(1,256);
                    keycodes(KbName({'l','z','q', 'space', 'Return'}))=1;
                    KbQueueCreate([], keycodes); %restrict to certain keys
                    KbQueueStart; %start recording keypresses
                end
                
                %RestrictKeysForKbCheck(KbName({'q','space'}));
                
                % Start trial
                trial.t0 = GetSecs(); %get time at beginning of trial
                
                % Start audio playback, return onset timestamp.
                PsychPortAudio('Start', pahandle, 1, 0, 1);
                
                % Wait until end of stimulus
                WaitSecs('UntilTime', trial.t0+size(trial.stim,2)/design.fs + design.endWait/1000);
                
                if ~design.passive
                    %% Record response time
                    %  the Psychtoolbox method of using 'StimulusOnsetTime' seems to be
                    %  the more reliable solution, specifically on varying hardware
                    %  setups or under suboptimal conditions
                    [trial.keypress.n, ~, ~, trial.keypress.t, ~] = KbQueueCheck();
                    
                    % If escape key (q) ---> ABORT
                    if trial.keypress.t(KbName('q')) > 0
                        trial.done=1;
                        break;
                    end
                    
                    % Restrict KbQueueCheck results to keys of interest (q/space)
                    trial.keypress.key = find(trial.keypress.t); %translate code into string
                    trial.keypress.t = trial.keypress.t(trial.keypress.key); %find RT of target button
                    trial.keypress.n = numel(trial.keypress.t);
                    
                    if ~isempty(trial.keypress.t)
                        trial.keypress.t = trial.keypress.t(1);
                        trial.keypress.key = trial.keypress.key(1);
                        trial.keypress.n = 1;
                    end
                    
                    %% Analyse keypress, evaluate response
                    [trial,results] = fAnalyseKey(trial,results,design);
                    
                    trial.keypress = []; % Erase keypress
                    Screen('Flip', expWin,0); % Flip screen
                    
                    KbQueueFlush;
                    %RestrictKeysForKbCheck(KbName({'z'}));
                    
                    
                    % Draw feedback on screen for Day 1: speed, accuracy
                    imdata=imread('_imgood.png');
                    imdata2=imread('_imbad.png');
                    im1=imread('_fast.png');
                    im2=imread('_slow.png');
                    im3=imread('_tooslow.png');
                    switch trial.correct
                        case 'Yes' %Correct response, green check mark
                            
                            if trial.respond == 0
                                Screen('PutImage',expWin,imdata);
                            elseif trial.respond == 1 && trial.ctr == 1
                                if trial.rt< design.rtedgesctr(1)
                                    Screen('PutImage',expWin,im1);
                                elseif trial.rt>design.rtedgesctr(1) && trial.rt<design.rtedgesctr(2)
                                    Screen('PutImage',expWin,im2);
                                elseif trial.rt>design.rtedgesctr(2)
                                    Screen('PutImage',expWin,im3);
                                end
                            elseif trial.respond == 1 && trial.ctr == 0
                                if trial.rt< design.rtedges(1)
                                    Screen('PutImage',expWin,im1);
                                elseif trial.rt> design.rtedges(1) && trial.rt< design.rtedges(2)
                                    Screen('PutImage',expWin,im2);
                                elseif trial.rt > design.rtedges(2)
                                    Screen('PutImage',expWin,im3);
                                end
                                %Screen('PutImage',expWin,Im);
                            end
                        case 'No' %Incorrect response, red cross
                            Screen('PutImage',expWin,imdata2);
                        case 'WrongKey' % DRAW BLUE
                            DrawFormattedText(expWin, 'Wrong key pressed.', 'center', 'center', 255, 0, [], [], 1.1);
                        case 'NoKey'  %No key was pressed, yellow fixation
                            DrawFormattedText(expWin, 'No key pressed.', 'center', 'center', 255, 0, [], [], 1.1);
                    end
                end
                
                
                Screen('Flip', expWin);
                % ISI
                WaitSecs  (design.ISI(randi(numel(design.ISI-design.endWait)))/1000); %wait ISI
                Screen('Flip', expWin,0); %
                
                
                % PRESS Q to BREAK
                currTime = 0;
                endTime = trial.t0+size(trial.stim,2)/design.fs + design.endWait/1000;
                while currTime < endTime
                    currTime = GetSecs();
                    [~,~,~,keypress] = KbQueueCheck();
                    if keypress(KbName('q')) >0
                        break
                    end
                end
                
                PsychPortAudio('DeleteBuffer');
                PsychPortAudio('Stop', pahandle);
            end
            
            
            
            %% Save data to file at end of each block - allows you to quit program
            if exist([outpath outfile '.mat'],'file')~=0 % check if file exists, in which case append x to name
                outfile = [outfile 'x'];
            end
            
            save([outpath outfile],'results','design','stimuli');
            
            
            if ~design.passive
                
                hits = sum(results.response==1);
                fpos = sum(results.response==-1)+sum(results.response==-2);
                fneg = sum(results.response == 0);
                crej = sum(results.response == 2);
                tgt_tot = hits + fneg;
                ntgt_tot = fpos + crej;
                RT_step = nanmean(results.RTs(results.condi==4));
                RT_randreg = nanmean(results.RTs(results.condi==2));
                dp = dprime(hits,fpos,tgt_tot,ntgt_tot,0);
                
                %                if design.whichBlock < design.nBlocks
                myText = ['End of block ' num2str(trial.currBlock) '\n'...
                    'Hits : ' num2str(hits) '\n' ...
                    'Misses : ' num2str(fneg) '\n' ...
                    'False Alarms : ' num2str(fpos) '\n' ...
                    '---(d prime : ' num2str(dp) ')---\n' ...
                    'RTs STEP : ' num2str(RT_step) '\n' ...
                    'RTs RANDREG : ' num2str(RT_randreg) '\n' ...
                    '\n'...
                    'End of block ' num2str(trial.currBlock) '\n'...
                    '(Call the experimenter and take a break)\n' ];
                
            else
                myText = ['End of block ' num2str(trial.currBlock) '\n'...
                    '(Call the experimenter and take a break)\n' ];
            end
            DrawFormattedText(expWin, myText, 'center', 'center', 255, 0, [], [], 1.1);
            Screen('Flip',expWin,1);
            KbQueueWait();
            trial.done=1;
            
            if trial.currBlock == design.nBlocks
                design.whichBlock = design.nBlocks+1;
            end
        end
        
    else
        
        %% start familiarity block
        if design.whichBlock > design.nBlocks &&  numel(design.familiarityBlock) == 1
            
            if design.whichBlock == design.nBlocks + 1
                b = 1;
            else
                b =2;
            end
            trial.currBlock = b; %make a loop if you want more familiarity blocks
            % Filename and path
            outfile = sprintf('log_sub%s_familiarity%d', design.subnum, b);
            outpath = design.resultsDir;
            rng(now+trial.currBlock);
            
            % Generate stimulus presentation order (should be randomised)
            condIDs=[];     stimIDs=[];
            for c= 1:numel(design.Fconds) %for all main conditions
                % Each condition is assigned a unique integer i between 1 and Nconds.
                % this corresponds to the order they are specified in Start_experiment
                % The reason why we want to balance stimulus numbers within each block
                % is because it makes it possible to compute d primes per block, and
                % also makes it easier to resume the experiment
                condIDs=[condIDs c*ones(1,design.FnEachCond(c)/design.FnBlocks)]; %condition IDs
                stimIDs =[stimIDs ...
                    stimuliF(c).order((trial.currBlock-1)*design.FnEachCond(c)/design.FnBlocks+1 : ...
                    trial.currBlock*design.FnEachCond(c)/design.FnBlocks)]; % stim IDs
            end
            
            shufOrder = randperm(design.FstimPerBlock); % uses same random order to reshuffle condition IDs and stimulus IDs
            design.FcondShuf{b} = condIDs(shufOrder);
            design.FstimShuf{b} = stimIDs(shufOrder);
            design.FnStimTot    = numel(condIDs)*design.FnBlocks;
            
            design.FnStimTot    = numel(condIDs)*design.FnBlocks;
            if ~design.passive
                design.rKey        = KbName('Space'); %response key is Space bar;
            end
            %% Set up a 'results' structure. This contains subjects results and stats
            results = [];
            results.condi       = NaN*ones(1,design.FstimPerBlock);   	%to  be filled in after each trial
            results.freqlist    = [];                       	%list of freqs used
            if ~design.passive
                results.response    = NaN*ones(1,design.FstimPerBlock);     %to  be filled in after each trial
                results.keypress    = NaN*ones(1,design.FstimPerBlock);   	%stores all keypresses
                results.RTs         = NaN*ones(1,design.FstimPerBlock);   	%reaction times
                results.correct     = NaN*ones(1,design.FstimPerBlock);
            end
            %% Set up a 'trial' structure. This contains all info pertaining to current trial
            trial.keypress  = [];
            
            if ~design.passive
                % Only record f8 and space, and enter keys
                keycodes=zeros(1,256);
                keycodes(KbName({'q', 'space', 'Return'}))=1;
                KbQueueCreate([], keycodes); % Restrict to certain keys
                KbQueueStart(); % Start recording keypresses
            end
            
            
            fixcross=Screen('MakeTexture',expWin,FixCr);
            myText2 = ['In this final block you are asked to say if you are familiar\n' ...
                'with some REGULAR patterns that were presented across the previous blocks\n' ...
                ' \n'...
                '>>> Press  <SPACEBAR> if you recognise a pattern \n' ...
                '>>> Do NOT press any button if the pattern is NOT familiar to you \n' ...
                '(Press <SPACEBAR> to continue)\n' ];
            
            DrawFormattedText(expWin, myText2, mx/2, 'center', 255, 0, [], [], 1.1);
            Screen('Flip', expWin);
            KbQueueWait;
            WaitSecs(1);
            
            for i=1:(design.FstimPerBlock)
                %% start trial
                trial.trialNum  = i;
                Screen('Flip', expWin);
                KbQueueFlush; %Empty keyboard queue before each trial
                
                Screen('DrawTexture', expWin, fixcross,[],[mx-10,my-10,mx+10,my+10]);
                Screen('Flip', expWin);
                
                % Gets the next trial; some of the trial parameters are stored in a
                % 'trial' struct
                [trial,results] = fnewTrial_fam(results,design,trial,stimuliF);
                trial.stim = [trial.stim; trial.stim]; % make it stereo
                
                % If recording EEG, add a trigger channel to the stimulus matrix
                if design.EEG
                    trial = fTrigger(trial,design);
                end
                
                % Fill the audio playback buffer with the audio data 'wavedata':
                PsychPortAudio('FillBuffer', pahandle, trial.stim);
                
                if ~design.passive
                    % Only record f8 and space, and enter keys
                    keycodes=zeros(1,256);
                    keycodes(KbName({'q', 'space', 'Return'}))=1;
                    KbQueueCreate([], keycodes); %restrict to certain keys
                    KbQueueStart; %start recording keypresses
                end
                
                % Start trial
                trial.t0 = GetSecs(); %get time at beginning of trial
                
                % Start audio playback, return onset timestamp.
                PsychPortAudio('Start', pahandle, 1, 0, 1);
                
                % Wait until end of stimulus
                WaitSecs('UntilTime', trial.t0+size(trial.stim,2)/design.fs + design.endWait/1000);
                
                if ~design.passive
                    %% Record response time
                    %  the Psychtoolbox method of using 'StimulusOnsetTime' seems to be
                    %  the more reliable solution, specifically on varying hardware
                    %  setups or under suboptimal conditions
                    [trial.keypress.n, ~, ~, trial.keypress.t, ~] = KbQueueCheck();
                    
                    % If escape key (q) ---> ABORT
                    if trial.keypress.t(KbName('q')) > 0
                        trial.done=1;
                        break;
                    end
                    
                    % Restrict KbQueueCheck results to keys of interest (f8/space)
                    trial.keypress.key = find(trial.keypress.t); %translate code into string
                    trial.keypress.t = trial.keypress.t(trial.keypress.key); %find RT of target button
                    trial.keypress.n = numel(trial.keypress.t);
                    trial.transit = 0;
                    %% Analyse keypress, evaluate response
                    [trial,results] = fAnalyseKey_fam(trial,results,design);
                    
                    trial.keypress = []; % Erase keypress
                    Screen('Flip', expWin,0); % Flip screen
                    
                    % Draw feedback on screen
                    switch trial.correct
                        %                         case 'Yes' %Correct response, green check mark
                        %                             imdata=imread('_imgood.png');
                        %                             Screen('PutImage',expWin,imdata);
                        %                         case 'No' %Incorrect response, red cross
                        %                             imdata=imread('_imbad.png');
                        %                             Screen('PutImage',expWin,imdata);
                        case 'WrongKey' % DRAW BLUE
                            DrawFormattedText(expWin, 'Wrong key pressed.', 'center', 'center', 255, 0, [], [], 1.1);
                        case 'NoKey'  %No key was pressed, yellow fixation
                            DrawFormattedText(expWin, 'No key pressed.', 'center', 'center', 255, 0, [], [], 1.1);
                    end
                end
                
                Screen('Flip', expWin);
                
                % ISI
                WaitSecs(design.ISI(randi(numel(design.ISI-design.endWait)))/1000); %wait ISI
                Screen('Flip', expWin,0); %
                
                % PRESS Q to BREAK
                currTime = 0;
                endTime = trial.t0+size(trial.stim,2)/design.fs + design.endWait/1000;
                while currTime < endTime
                    currTime = GetSecs();
                    [~,~,~,keypress] = KbQueueCheck();
                    if keypress(KbName('q')) >0
                        break
                    end
                end
                
                PsychPortAudio('DeleteBuffer');
                PsychPortAudio('Stop', pahandle);
                
            end
            
            
            %% Save data to file at end of each block - allows you to quit program
            if exist([outpath outfile '.mat'],'file')~=0  % check if file exists, in which case append x to name
                outfile = [outfile 'x'];
            end
            save([outpath outfile],'results','design','stimuliF');
            
            
            if ~design.passive
                hits = sum(results.response==1);
                fpos = sum(results.response==-1);
                fneg = sum(results.response == 0);
                crej = sum(results.response == 2);
                tgt_tot = hits + fneg;
                ntgt_tot = fpos + crej;
                % dp = dprime(hits,fpos,tgt_tot,ntgt_tot,0);
                
                myText = ['End of block \n'...
                    'Hits : ' num2str(hits) '\n' ...
                    'Misses : ' num2str(fneg) '\n' ...
                    'False Alarms : ' num2str(fpos) '\n' ];
                %                 '---(d prime : ' num2str(dp) ')---\n' ...
                %                 '(Press any key to continue)\n' ];
            else
                myText = ['End of block \n'...
                    '(Press any key to continue)\n' ];
            end
            DrawFormattedText(expWin, myText, 'center', 'center', 255, 0, [], [], 1.1);
            Screen('Flip',expWin,1);
            KbQueueWait();
            trial.done=1;
            
        end
    end
    
    if trial.done
        myText = [' \n' ];
        
        DrawFormattedText(expWin, myText, 'center', 'center', 255, 0, [], [], 1.1);
        Screen('Flip',expWin,1);
        KbQueueWait();
    end
    fCleanup();
    
catch err
    % This section is executed only in case an error happens in the
    % experiment code implemented between try and catch...
    fCleanup();
    rethrow(err)
end
%% Finally put the main variable in the workspace (for debugging)
assignin('base', 'design', design);
assignin('base', 'results', results);
assignin('base', 'trial', trial);
clc
end
