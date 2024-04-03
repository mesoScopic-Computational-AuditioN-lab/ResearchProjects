function fct=fgetLoudnessCorrectionFactor(freq)

% Pure tones (PT) %%%%
% disp('Loudness calculation for pure tones (ISO 532B)...');

% 1) calculateloudness at phon 60 
% FYI phon = SPL(in db) of a 1-kHz pure tone judged equally loud
dB=iso226(60,freq); %using the 60 phon baseline

% 2) Then if the loudness hasn't changed, the removal of 60 should results in 0. If not you have the difference in loudness. 
dbDiff=dB-60;


fct=10^(dbDiff/20);  %power_db = 20 * log10(amp / amp_ref);
end
