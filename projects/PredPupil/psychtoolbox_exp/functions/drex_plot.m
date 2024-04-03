function drex_plot(pres_freq, n_run, n_block)

% Make N by 2 matrix of fieldname + value type
variable_names_types = [["frequencies", "double"]; ...
			["frequencies_oct", "double"]; ...
			["run", "double"]; ...
            ["block", "double"]; ...
			["segment", "double"]];
        
% Make table using fieldnames & value types from above
new_data = table('Size',[0,size(variable_names_types,1)],... 
	'VariableNames', variable_names_types(:,1),...
	'VariableTypes', variable_names_types(:,2));

for r=1:n_run
    for b = 1:n_block
        
        data = pres_freq(r,b,:);
        frequencies     = squeeze(data);
        frequencies_oct = log2(frequencies);
        block           = ones(size(pres_freq,3),1)*b;
        run             = ones(size(pres_freq,3),1)*r;
        segment         = vertcat(ones(20,1), ones(48,1)*2);

        t = table(frequencies, frequencies_oct, run, block, segment,... 
            'VariableNames', variable_names_types(:,1));
        new_data = [new_data;t];

    end
end

%% run drex
addpath('/Users/scanlab/Documents/internship_luca/model-folder/drex-model/')

params = [];
params.distribution = 'gmm';
params.max_ncomp = 2;
params.beta = 0.2; % scale down with higher value - probably
params.D = 1;

params.maxhyp = inf;
params.memory = inf;

% go through each block
for rn = 2:n_run
for blk = 1:n_block
    idxs = find(new_data.block==blk & new_data.run == rn);

    x = new_data.frequencies_oct(idxs(1):idxs(end));

    params.prior = estimate_suffstat(x,params);

    out = run_DREX_model(x,params);

    new_data.drex_surp(idxs(1):idxs(end)) = out.surprisal;

    % display figure of model
    figure(blk); clf;
    display_DREX_output(out,x)
end
end

return