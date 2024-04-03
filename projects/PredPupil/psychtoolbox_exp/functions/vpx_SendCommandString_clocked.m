function [clk, command_delay] = vpx_SendCommandString_clocked(str)
%-------------------------------------------------------
%vpx_SendCommandString
%  vpx_SendCommandString provide a mechanism to 
%	  send CLP command strings to ViewPoint.
%
%   USAGE: [r]=vpx_SendCommandString(str);
%   INPUT: str (The command to be sent to ViewPoint
%   OUTPUT: r, clock
%
%   ViewPoint EyeTracker Toolbox (TM)
%   Copyright 2005-2010, Arrington Research, Inc.
%	All rights reserved.
%--------------------------------------------------------
strfinal=strcat([str,blanks(1),'//']);
clk = GetSecs;
calllib('vpx', 'VPX_SendCommandString',char(strfinal));
command_delay = GetSecs-clk;