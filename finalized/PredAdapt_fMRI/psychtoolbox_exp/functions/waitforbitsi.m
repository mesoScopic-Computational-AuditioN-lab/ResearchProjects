function response = waitforbitsi(pulse, esckey, shiftkey, B, varargin)

% make sure if we quit we can save everything we want
% note this is NOT recommended (dynamically naming variables)
varz = {};
for var = 1:nargin-4
    varz{var} = inputname(var+4);
    eval([inputname(var+4) '= varargin{var};']);
end

% set start response to 0
response = 0;

% wait for bitsi
while 1
    [a,~,c] = KbCheck;
    if response == 0 && a && sum(ismember(pulse,find(c))) > 0
        response = c;
        break
    elseif a && ismember(esckey,find(c)) && sum(ismember(shiftkey,find(c))) > 0
        save (fullfile( pwd, 'data', 'TEMP-Save.mat'), varz{:});
        sca; ShowCursor;
        error('[!!!] Program aborted by user');
    elseif not(B.debugmode) && response == 0 && B.serobj.BytesAvailable > 0
        response = fread(B.serobj, 1);
        % allow only certain triggers
        if sum(ismember(pulse,find(response))) > 0
            break
        else
            response = 0;
        end
    end
end