function tmpresp = waitforbitsi_backup(pulse, cfg, varargin)%waitforbitsi_backup(pulse, esckey, shiftkey, cfg, varargin)

% make sure if we quit we can save everything we want
% note this is NOT recommended (dynamically naming variables)
varz = {};
for var = 1:nargin-2
    varz{var} = inputname(var+2);
    eval([inputname(var+2) '= varargin{var};']);
end

% wait for bitsi
cfg.setup.B.validResponses = pulse;
[tmpresp, ~, ~] = cfg.setup.B.getResponse(inf, 1);
if (tmpresp == 0)   % if timed out
    tmpresp = nan;
else 
    tmpresp = tmpresp(1);
end

end
