function save_clean_timings(ppnum, stimlen, padding)
% load timings of pp and clean up using overshoot info and padding info

ppnum = num2str(ppnum);

load (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz');
timingz_cleaned = clean_timings(timingz,stimlen,padding);
save (fullfile( pwd, 'data', ppnum, [ppnum '-mainpred.mat']), 'responses', 'segmentz', 'timingz', 'timingz_cleaned');

end