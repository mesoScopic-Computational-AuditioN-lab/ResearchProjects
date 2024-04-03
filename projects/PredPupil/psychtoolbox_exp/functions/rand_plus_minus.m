function num2ret = rand_plus_minus(num)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                  %
%   RETURN NUMBER WITH + OR - RANDOMLY                             %
%     Input:                                                       %
%       - num: number to be applied on                             %
%     Returns:                                                     %
%       - num2ret: number with random sign applied on              %
%                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

signs = [-1 1];

num2ret = num*(signs(randi(2)));

return