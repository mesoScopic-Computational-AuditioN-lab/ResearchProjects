function dp = dprime(hit,fa,targ, notarg,correction)

% calculate d prime

% hit = number of hits fa = number of false alarms targ = numebr of trials
% with target notarg = number of trials without target 
% correction - what
% type of correction for when hit rate or false alarm rate is maximal or
% minimal


hr=hit/targ;
fr=fa/notarg;



switch correction
    case 0 % subtract/add half a trial before calculating rate
        if hr ==1
            hit = hit-0.5;
            hr = hit/targ;
        end
        if fr ==0
            fa=0.5;
            fr=fa/notarg;
        end
        if hr ==0
            hit = 0.5;
            hr = hit/targ;
        end
        if fr ==0
            fa=fa+0.5;
            fr=fa/notarg;
        end
    case 1
        % subtract/add half a % from hit rate / fa rate once calculated
        % (appropriate for correcting control condition when comparing to main
        % conditions which had a greater number of observatins - otherwise we get a
        % d prime lower for perfect performance in one condition than another with
        % imperfect performance!
        if hr==1
            hr=0.995;
        end
        if fr==0
            fr = 0.005;
        end
end

dp = norminv(hr)-norminv(fr);

if targ == 0 || notarg == 0 % we cant calc d' if we dont have both types of trial!
    dp = NaN;
end