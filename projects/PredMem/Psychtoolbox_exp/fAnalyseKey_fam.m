function [tr,res] = fAnalyseKey_fam(tr,res,des)
%res.stimulus{tr.trialNum}= tr.stim;
keypress = tr.keypress;
n = keypress.n;
tr.rt = NaN;
if n>0
    key=keypress.key(end); % accept last button press in case of multiple response
else
    key=NaN;
end
res.keypress(tr.trialNum)=key;

if des.condi(1) == 99 % i.e. if its the control RT task
    if isnan(key)
        tr.rt=NaN;
        tr.correct = 'No';
        res.correct(tr.trialNum) = 0;
        res.response(tr.trialNum) = 0;
        
    else
        tr.rt = keypress.t - tr.t0;
        tr.correct = 'Yes';
        res.correct(tr.trialNum) = 1;
        res.response(tr.trialNum) = 1;
        
    end
    res.RTs(tr.trialNum)=tr.rt; % Reaction time
    
else
    
%     % calculate deviant tone onset time
%     
%     tr.dev_t = tr.dev_i * des.toneDur /1000;
%     res.nkeys(tr.trialNum)=n;

    tr.expekey = des.toneDur * tr.transit/1000;
    res.nkeys(tr.trialNum)=n;  
    
    % Interpret the response provide feedback and deal with
    res.target(tr.trialNum) = tr.respond;
    res.condi(tr.trialNum) = tr.condi;
    res.setID(tr.trialNum) = tr.setID;
    res.expekey(tr.trialNum) = tr.expekey;
    
    if isnan(key)  %No key was pressed
        if tr.respond == 0 % No Target, No Key press: CR, green fixation
            res.response(tr.trialNum) = 2;
            res.correct(tr.trialNum) = 1;
            tr.correct = 'Yes';
        elseif tr.respond == 1 % Miss, red fixation
            res.response(tr.trialNum) = 0;
            res.correct(tr.trialNum) = 0;
            tr.correct = 'No';
        end
        
    else % Key was pressed

%                 % __________RT__________
%         if tr.dev
%             tr.rt = keypress.t - tr.dev_t - tr.t0;
%         else % if no deviant, calculate RT relative to start of stimulus
%             tr.rt = keypress.t - tr.t0;
%         end

        tr.rt = keypress.t - tr.t0;


        % __________code response type_________
        if key==des.rKey && tr.respond == 1 && tr.rt>0% Hit, green fixation
            res.response(tr.trialNum) = 1;
            res.correct(tr.trialNum) = 1; 
            tr.correct = 'Yes';
        elseif key==des.rKey && tr.respond  == 0 % FA, red fixation
            res.response(tr.trialNum) = -1;
            res.correct(tr.trialNum) = 0;
            tr.correct = 'No';
        elseif key ~= des.rKey %Wrong key was pressed
            res.response(tr.trialNum) = NaN;
            tr.correct = 'WrongKey';
            res.correct(tr.trialNum) = NaN;
        end
        
        % Update results structure with performance at current trial
        res.RTs(tr.trialNum)=tr.rt; % Reaction time
        res.t0{tr.trialNum} = tr.t0;
        res.tkey{tr.trialNum} = keypress.t;
     end
end
end