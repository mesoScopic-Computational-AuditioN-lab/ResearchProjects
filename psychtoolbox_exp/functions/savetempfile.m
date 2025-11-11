function savetempfile(ppnum, wrun)

ppnum = num2str(ppnum);
wrun = num2str(wrun);

load (fullfile( pwd, 'data', ['TEMP-Save.mat']), 'responses', 'segmentz', 'timingz', 'pulsez');
save (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz');
save (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred(backup).mat']), 'responses', 'segmentz', 'timingz');
save (fullfile( pwd, 'data', ppnum, ['_' ppnum '-r' wrun '-pulses.mat']), 'pulsez');
save (fullfile( pwd, 'data', ppnum, ['_' ppnum '-r' wrun '-pulses(backup).mat']), 'pulsez');

end