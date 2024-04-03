function [pix,oslu,pplu] = ang2pix(vangle,dispdist,dispwidth,disppix,rounder) 
%% 
% Converts degrees of visual angle into number of pixels
%
%
% INPUT:
%
% - Visual angle in degrees 
% - Distance between the observer and the display
% - Width or height of the display in length units (e.g., cm or inches)
% - Width or height of the display in number of pixels
%
% - Optional: function rounds to whole pixels unless fed "0"
%
%
%
% OUTPUT: 
%
% - Stimulus size in number of pixels
%
% - Optional: object size in length units
% - Optional: number of pixels per length unit
%
% -----------------------------------------------------------------------_

oslu=(2*dispdist*tand(0.5*vangle)); % oslu = object size in length units
pplu=disppix/dispwidth; % pplu = number of pixels per length unit
if nargin==4||rounder~=0;
   pix=round(oslu*pplu);
else
    pix = oslu*pplu;
end

 




