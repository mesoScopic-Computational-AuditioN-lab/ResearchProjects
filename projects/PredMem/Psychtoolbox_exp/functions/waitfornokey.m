function [] = waitfornokey
while 1
    [a,~,~] = KbCheck;
    if a ==0;
        break
    end
end