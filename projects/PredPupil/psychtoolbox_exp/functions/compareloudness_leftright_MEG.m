function [relloudness, cc] = compareloudness_leftright_MEG(soundmatrix,equal, w, screenrect)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                      %
%      MAIN EQUALISATION TASK, COMPARISON ON TWO TONES IN LOUDNESS     %
%                                                                      %                                                                      %%
%    Input:                                                            %
%      - refstim: [lengthstim x 1] sound sequence of refference        %
%      - compstim: [lengthstim x loudnesses] sequence of target        %
%      - equal: struct with settings for equalistation                 %
%      - [w, screenrect]: psychtoolbox drawscreen                      %
%                                                                      %
%    Return:                                                           %
%      - relloudness: relative loudness value compared to refference   %
%      - cc: last pressed button                                       %
%                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert needed inputs into variables
ref_loud    =  equal.refloud;
ind_loud    =  equal.ind_loud;
samp_rate   =  equal.samp_rate;
sil_gap     =  equal.sil_gap;
out         =  0; % for while loop

% bitsi initialize
b = Bitsi('com1');

% Perform basic initialization of the sound driver:
PsychPortAudio('Close'); 
InitializePsychSound(1);
pahandle = PsychPortAudio('Open', [], 1, 1, samp_rate, equal.nrchannels, [], 1);

% load fixation bull into memory
BullTex(1)     = get_bull_tex_2(w, equal.bull_eye_col, equal.bull_in_col, equal.bull_out_col, equal.bull_fixrads);
bullrect       = CenterRect([0 0 equal.bullsize equal.bullsize], screenrect);

% display intro text
multilinetext(w, equal.introtxt, screenrect, ...
              equal.fontsize, equal.textfont, equal.textcol, 1.2, [3]);
Screen('Flip', w); b.getResponse(inf, true);

% wait for key stroke
multilinetext(w, equal.keytxt, screenrect, ...
              equal.fontsize, equal.textfont, equal.textcol, 1.2, [5]);
Screen('Flip', w); b.getResponse(inf, true);

% main equalisation procedure
while out~=1
    
    % draw fixation
    Screen('DrawTexture',w,BullTex(1),[],bullrect);
    [~] = Screen('Flip', w);
    
    % build stim
    stim_r = soundmatrix(ref_loud, :);
    stim_r = [stim_r, zeros(1,(samp_rate*sil_gap)+length(stim_r))];       % add silence gap and other channel

    stim_l = soundmatrix(ind_loud, :);
    stim_l = [zeros(1,(samp_rate*sil_gap)+length(stim_l)), stim_l];         % add silence gap and other channel

    % loud and play audio
    PsychPortAudio('FillBuffer', pahandle, [stim_r ; stim_l]);
    PsychPortAudio('Start', pahandle, 1, 0, 1);
    
    % wait for key stroke
    multilinetext(w, equal.keytxt, screenrect, ...
                  equal.fontsize, equal.textfont, equal.textcol, 1.2, [5]);
    WaitSecs(2);
    Screen('Flip', w);% Show the drawn text at next display refresh cycle
    
    %wait for input
    [keyCode, ~] = b.getResponse(inf, true);
    
    %calculate performance or detect forced exit
    if keyCode == 104 % if redleft
        sca; 
        break; 
    elseif keyCode == 97
        ind_loud = ind_loud+1;
        ind_loud = min(ind_loud,size(soundmatrix,1));
    elseif keyCode == 98
        ind_loud = ind_loud-1;
        ind_loud = max(ind_loud,1);
    elseif keyCOde == 99
        out = 1;
    end
    
end
relloudness = ind_loud;     % save new relative loudness
PsychPortAudio('Close', pahandle);% Close the audio device: