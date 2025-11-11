function b = waitforpulse(pulse, esckey, shiftkey, varargin)

% make sure if we quit we can save everything we want
% note this is NOT recommended (dynamically naming variables)
varz = {};
for var = 1:nargin-3
    varz{var} = inputname(var+3);
    eval([inputname(var+3) '= varargin{var};']);
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