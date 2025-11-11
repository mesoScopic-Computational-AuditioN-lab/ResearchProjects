function timingz_cleaned = clean_timings(timingz, stimlen, padding)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                    %
%     CLEAN UP TIMING DATA FROM MAIN EXPERIMENT, USES AVALABLE       %
%      TIMING DATA (BUFFER PLAY OVERSHOOT DATA) AND LENGTH OF        %
%      PADDING TO CALCULATE WHEN ACTUAL AUDIO ONSET OCCURED          %
%                                                                    %
%     INPUT: timingz data, stimulus length, and padding length(ms)   %
%     OUTPUT: timingz data with correct timing                       %
%                                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

padding = 0.01; % what padding settings were used
stimlen = 0.2;  % what was the stimlength used

% calculate length of what supposed to be in the buffer
buflen = padding+stimlen;
bufspil = zeros(1, size(timingz(timingz(2,:)==1),2));         % spilover because of previous under/overshoot

% new cleaned timingz matrix
timingz_cleaned               =   nan(8, size(timingz,2));
timingz_cleaned(1:4, :)       =   timingz(1:4,:);             % copy important trial info
timingz_cleaned(  5, :)       =   nan;                        % start time of actual sound
timingz_cleaned(  6, :)       =   nan;                        % stop time of actual sound
timingz_cleaned(  7, :)       =   nan;                        % length of actual sound presentation
timingz_cleaned(  8, :)       =   nan;                        % buffer stop over/undershoot (minus=undershoot, plus=overshoot target endpoint)
timingz_cleaned(  9, :)       =   nan;                        % startplay buffer spilover from previous pres

% loop over blocks (since it was buffered per block) and calculate endpos error 
for block = 1:max(timingz(2,:))
    bufferundershoot                                        =   diff([0 , timingz(10, timingz(2,:) == block)]) - buflen;        % calculate over and undershoots of buffered audio
    timingz_cleaned(8, timingz_cleaned(2,:) == block)       =   bufferundershoot;                                               % save buffer over/undershoot info
    bufspil(2:end)                                          =   bufferundershoot(1:end-1);                                      % shift one to see how mutch extra/less time we presented of next snipped
    timingz_cleaned(9, timingz_cleaned(2,:) == block)       =   bufspil;                                                        % also save this information
end

timingz_cleaned(  5, :)       =   timingz(7,:) + timingz_cleaned(9,:) + (padding/2);                                            % calculate actual starttime by using the over/undershoot and padding length
timingz_cleaned(  7, :)       =   (stimlen*timingz(9,:))/buflen;                                                                % calculate actual presentation length (using how the audio driver can go out of sinc)
timingz_cleaned(  6, :)       =   timingz_cleaned(5,:)+timingz_cleaned(7,:);                                                    % calculate actual end of presentation time

return