function print_tones_block_by_block(tones, nr_runs, nr_blocks, bpr, condNames)
    %% print_tones_block_by_block
    %   
    %	Displays tones in selected number of blocks and runs in log2
    %   dimension, together with the mean of the tones in each segment.
    %
    %   USAGE 
    %		print_tones_block_by_block(tones, nr_runs, nr_blocks, bpr, condNames);
    %
    %   IMPORTANT: 
    %       - displays tones in selected number of blocks and runs in log2
    %       dimension, together with the mean of the tones in each segment
    %   
    %------------------------------------------------------
    
    arguments
        tones cell;
        nr_runs double;
        nr_blocks double;
        bpr double;
        condNames cell;
    end

    for r = 1:nr_runs
    for b = 1:nr_blocks

        % set data and mean
        s1_idx = 1+2*(b-1) + (bpr*(r-1));
        s2_idx = 2+2*(b-1) + (bpr*(r-1));
        s1_mean = mean(log2(tones{s1_idx}));
        s2_mean = mean(log2(tones{s2_idx}));

        % define some variables
        data_len = size(tones{s1_idx},2) + size(tones{s2_idx},2);
        data = zeros(1,data_len);
        data(1:size(tones{s1_idx},2)) = tones{s1_idx};
        data(size(tones{s1_idx},2)+1:end) = tones{s2_idx};
        mean_data = zeros(1,data_len);
        mean_data(1:size(tones{s1_idx},2)) = s1_mean;
        mean_data(size(tones{s1_idx},2)+1:end) = s2_mean;

        % set condition description
        cond_descr = join([condNames(1+(2*(b-1))+(nr_blocks*(r-1))) '-' condNames(2+(2*(b-1))+(nr_blocks*(r-1)))]);

        % plotting
        subplot(nr_runs,nr_blocks,b+(nr_blocks*(r-1))); 
        text = join(["Run: " num2str(r) " | Block" num2str(b) '|' cond_descr]);
        scatter(1:size(data,2),log2(data),'filled')
        hold on
        plot(1:size(mean_data,2),mean_data)
        
        hold off
        title(text)

    end
    end

return