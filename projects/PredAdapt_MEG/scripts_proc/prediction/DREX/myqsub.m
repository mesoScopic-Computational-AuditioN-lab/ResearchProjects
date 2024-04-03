function myqsub(funname, reqstring, varargin)
% MYQSUB allows easy submission of many jobs to a Torque cluster via qsub.
% funname specifies the function to execute, reqstring is the requirements
% (in a format that qsub understands as an argument to the "-l" switch).
% Any subsequent arguments can only be numeric and will be interpreted as
% arguments of the function specified by funname.
%
% For example, if you have a function with the signature
% run_freq_analysis_single_subj(subj_id, low_or_high_freqs), then you could
% run this code for all subjects, low frequencies only, on the cluster as
% follows (assumng subject IDs run from 1 to 34):
% >> myqsub('run_freq_analysis_single_subj', 'walltime=00:20:00,mem=6gb',
%           1:34, zeros(34,1));
% assuming furthermore that a 0 for the input argument low_or_high_freqs means
% that the function should do the low frequencies.
%
% Put this utility function somewhere on your path, and optionally customize the path
% variables inside it. There are three customizable paths: (1) the Matlab
% working directory, (2) the Matlab executable, and (3) the log file
% directory. These are hard-coded into the function on purpose, rather than
% specified as arguments, in order to keep the calling syntax lean. By
% default the same Matlab is chosen as the one you use to execute myqsub,
% working directory is the one you are in when calling myqsub, and log files
% are stored in <HOME>/.matlabjobs.
%
% Copyright Eelke Spaak, 2018. Licensed under Apache License 2.0.

% which matlab executable too use for the jobs
%matlabcmd = sprintf('%s/bin/matlab -nodesktop -nosplash', getenv('MATLAB'));
matlabcmd = '/opt/matlab/R2023a/bin/matlab -nodesktop -nosplash';

% in which working directory to start the jobs
%workingdir = pwd();
workingdir = '/project/3018063.01/scripts_proc/prediction/DREX/';

% where to store the log files? per batch, a subdirectory will be created
% here
logdir = [workingdir '/qsubjobs/'];%'~/.matlabjobs';

if strcmp(varargin{end}, '-noquit')
  noquit = 1;
  varargin = varargin(1:end-1);
else
  noquit = 0;
end

nargs = numel(varargin);
njob = numel(varargin{1});

% generate a batch identifier, used in naming the log files
batchid = datestr(datetime(), 'yyyy-mm-ddTHH-MM-SS');
mkdir(sprintf('%s/%s', logdir,  batchid));

for k = 1:njob
  args = {};
  for l = 1:nargs
    args{l} = num2str(varargin{l}{k});
  end
  args = join(args, ',');

  if noquit
    matlabscript = sprintf('cd %s; %s(%s);', workingdir, funname, args{1});
  else
    matlabscript = sprintf('cd %s; %s(%s); quit', workingdir, funname, args{1});
  end

  % store the output in custom files
  logfile = sprintf('%s/j%s_%s', batchid, args{1}, funname);
  
  qsubcmd = sprintf('qsub -q matlab -l %s -N j%s_%s', reqstring, args{1}, funname);
  cmd = sprintf('echo ''stdbuf -oL -eL %s -r "%s" >%s/%s.out 2>%s/%s.err'' | %s',...
    matlabcmd, matlabscript, logdir, logfile, logdir, logfile, qsubcmd);
  
  %fprintf('%s\n', cmd);
  [status, result] = system(cmd);
  if status
    error(result);
  end
end

end