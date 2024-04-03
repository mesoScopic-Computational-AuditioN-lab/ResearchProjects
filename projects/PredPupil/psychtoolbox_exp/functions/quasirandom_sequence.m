function sequence_order = quasirandom_sequence(sequence_length, leap, temporal_mem)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   QUASI-RANDOM PERMUTATION GENERATOR WITH MIN. SET LEAPS BETWEEN STEPS
% 
%   Input:
% 
%   - sequence_length:    length of sequence range
%   - leap:               minimum step distance between permutation steps
%   - temporal_mem:       temporal memory of leap 
%                         (how many steps should leap take into account)
% 
%   Output:
% 
%   - sequence_order:     the adjusted, quasi-random sequence order
% 
%   Jorie van Haren (2022)                                            v1.0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% START MAIN FUNCTION
count = 0;                  % set count of tries
err_count = 0;              % set error count to count failed tries
while count == err_count 
try                         % try to generate sequence

input_range         = linspace(1, sequence_length, sequence_length);    % set input range
sequence_order      = nan(1, sequence_length);                          % predefine output range

sequence_order(1)   = randsample(input_range,1);                        % set initail value
input_range(input_range == sequence_order(1)) = [];                     % delete initial value from order

for i = 2:sequence_length                                               % loop over sequence

    % make sure we can sample the last value 
    % (it must be an array and not an int)
    if size(input_range, 2) == 1                                                  
        input_range = [input_range input_range];
    end

    differences = diff(input_range) == min(diff(input_range));         % calculate distances between successors in seq.
    differences(end+1) = differences(end);                             % copy last to match length

    % count back to first row and not further
    if i <= temporal_mem
        mem = i-1;
    else
        mem = temporal_mem;
    end

    % loop over temporal memory range and set leap_range
    leap_range = ones(1, length(input_range));
    for t = 1:mem
        leap_range = and(leap_range, (abs(input_range-sequence_order(i-t)) >= leap));
    end

    % check if our size of possibilities is 1
    if size(input_range(leap_range & differences), 2) == 1
        cur_samp = input_range(leap_range & differences);            % if only one pick that option
    else
        if size(input_range(leap_range & differences)) > 0          % check if any possibilities
            cur_samp = randsample(input_range(leap_range & differences),1);  % sample from possibilities
        else
            cur_samp = randsample(input_range(leap_range),1);           % or sample from bigger subsample
        end
    end

    sequence_order(i) = cur_samp;                   % save current sample in output sequence
    input_range(input_range == cur_samp) = [];      % delete from input sequence

end

if ~all(diff(sort(sequence_order)))                 % check if anything wierd happend while matching (if output order is ok)
    error('count mismatch')
end

catch                                               % catch errors (for inpossible sequences and errorous sequences)
    err_count = err_count + 1;                      % increase error count
    rng('default');

    % if count is to high abort
    if err_count == 1
        warning('Pseudorandom sequence stalled... retrying')
    elseif err_count == 200
        warning('Taking longer then expected, are input parameters possible?')
    elseif err_count > 1000
        error('Program stalled, check if input parameters are possible and try again')
    end

end

count = count + 1;                                  % increase loop count
end

return
