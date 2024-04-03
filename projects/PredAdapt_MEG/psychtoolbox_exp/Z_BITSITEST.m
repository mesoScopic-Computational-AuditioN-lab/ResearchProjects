% TESTING BITSI / KEYBOARD MULTI FUNCTION

clear mex
addpath(genpath([pwd '/functions/'])); startup1;

% setup bitsi
B = Bitsi('');
% B = Bitsi('com1');

% set keyboard values
pulse     = [32 97 98 99 100 101 102 103 104];
shiftkey  = [160 161];
esckey    = 27;

% set start response to 0
response = 0;

% wait for bitsi
while 1
    [a,~,c] = KbCheck;
    if response == 0 && a && sum(ismember(pulse,find(c))) > 0
        response = find(c);
        disp(['RESPONSE: ' num2str(response)]); %TEMPFORTESTING
        break
    elseif a && ismember(esckey,find(c)) && sum(ismember(shiftkey,find(c))) > 0
%         save (fullfile( pwd, 'data', 'TEMP-Save.mat'), varz{:});
        disp(['RESPONSE: ' num2str(find(c))]); %TEMPFORTESTING
        sca; ShowCursor;
        error('[!!!] Program aborted by user');
    elseif not(B.debugmode) && response == 0 && B.serobj.BytesAvailable > 0
        response = fread(B.serobj, 1);
        disp(['RESPONSE: ' num2str(response)]); %TEMPFORTESTING
        % allow only certain triggers
        if sum(ismember(pulse,find(response))) > 0
            break
        else
            response = 0;
        end
    end
end