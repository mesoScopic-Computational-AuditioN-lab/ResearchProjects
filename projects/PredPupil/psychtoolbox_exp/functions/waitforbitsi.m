function response = waitforbitsi(key, esckey, shiftkey, varargin)

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
    if response == 0 && a && sum(ismember(key,find(c))) > 0
        response = c;
        break
    elseif a && ismember(esckey,find(c)) && sum(ismember(shiftkey,find(c))) > 0
        save (fullfile( pwd, 'data', 'TEMP-Save.mat'), varz{:});
        sca; ShowCursor;
        error('[!!!] Program aborted by user');
    end
end

% wait for keyboard
while 1
    [a,b,c] = KbCheck;
    if a && sum(ismember(pulse,find(c))) > 0
        break
    elseif a && ismember(esckey,find(c)) && sum(ismember(shiftkey,find(c))) > 0
        save (fullfile( pwd, 'data', 'TEMP-Save.mat'), varz{:});
        sca; ShowCursor;
        error('[!!!] Program aborted by user');
    end
end