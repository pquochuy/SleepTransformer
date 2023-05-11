function [ret] = process_and_save_1file(filename, n)
    xml_path = '~/Dataset/shhs/polysomnography/annotations-events-nsrr/shhs1/';
    edf_path = '~/Dataset/shhs/polysomnography/edfs/shhs1/';
    mat_path = './mat/';
    
    ret = 1;
    
    if(exist([mat_path, 'n', num2str(n,'%04d'),'_eeg.mat'], 'file') && ...
        exist([mat_path, 'n', num2str(n,'%04d'),'_eog.mat'], 'file') && ...
        exist([mat_path, 'n', num2str(n,'%04d'),'_emg.mat'], 'file'))
        return;
    end

    fs = 100;

    epoch_second = 30;
    win_size  = 2;
    overlap = 1;
    nfft = 2^nextpow2(win_size*fs);
    
    [~,name,ext] = fileparts(filename);
    stages = read_shhs_annotation([xml_path, name, '-nsrr.xml']);
    
    [eeg, ori_fs] = read_shhs_edfrecords([edf_path, filename], {'EEG'});
    if(ori_fs ~= fs) % resampling
        eeg = resample(eeg, fs, ori_fs);
    end
    % preprocessing filter
    Nfir = 100;
    b_band = fir1(Nfir,[0.3 40].*2/fs,'bandpass'); % bandpass
    eeg = filtfilt(b_band,1,eeg);
    
    [eog, ori_fs] = read_shhs_edfrecords([edf_path, filename], {'EOGL','EOGR'});
    if(ori_fs ~= fs) % resampling
        eog = resample(eog, fs, ori_fs);
    end
    % preprocessing filter
    Nfir = 100;
    b_band = fir1(Nfir,[0.3 40].*2/fs,'bandpass'); % bandpass
    eog = filtfilt(b_band,1,eog);
    
    [emg, ori_fs] = read_shhs_edfrecords([edf_path, filename], {'EMG'});
    if(ori_fs ~= fs) % resampling
        emg = resample(emg, fs, ori_fs);
    end
    % preprocessing filter
    Nfir = 100;
    %pwrline = 50; %Hz
    %b_notch1 = fir1(Nfir,[(pwrline-1) (pwrline+1)].*2/fs,'stop');
    %emg = filtfilt(b_notch1,1,emg);
    %pwrline = 60; %Hz
    %b_notch2 = fir1(Nfir,[(pwrline-1) (pwrline+1)].*2/fs,'stop');
    %emg = filtfilt(b_notch2,1,emg);
    b_band = fir1(Nfir,10.*2/fs,'high'); % highpass
    emg = filtfilt(b_band,1,emg);
    
    assert(length(eeg)/(epoch_second*fs) == numel(stages))
    assert(length(eog)/(epoch_second*fs) == numel(stages))
    assert(length(emg)/(epoch_second*fs) == numel(stages))
    
    eeg_epochs = buffer(eeg, epoch_second*fs);
    eeg_epochs = eeg_epochs';
    eog_epochs = buffer(eog, epoch_second*fs);
    eog_epochs = eog_epochs';
    emg_epochs = buffer(emg, epoch_second*fs);
    emg_epochs = emg_epochs';
    
    ind = find(stages > 5 | stages < 0);
    if(sum(ind) > 0)
        disp([filename, ': ', num2str(sum(ind)), ' UKNOWN epochs removed']);
        eeg_epochs(ind, :) = [];
        eog_epochs(ind, :) = [];
        emg_epochs(ind, :) = [];
        stages(ind) = [];
    end
    
    % original labels: Wake (0), N1(1), N2(2), N3(3), N4(4), REM(5)
	% verify that there is no other stage
    [labels] = unique(stages);
	disp(labels)
	assert (max(labels) <= 5)
    % remove other stages
    
    
    count_stage = hist(stages,labels);
    if(count_stage(1) > max(count_stage(2:end))) % if too much W
        disp('Wake is the biggest class. Trimming it..')
        second_largest = max(count_stage(2:end));
        
        W_ind = (stages == 0); % W indices
        last_evening_W_index = find(diff(W_ind) ~= 0, 1, 'first')+1;
        if(stages(1) == 0) % only true if the first epoch is W
            num_evening_W = last_evening_W_index;
        else
            num_evening_W = 0;
        end
        
        first_morning_W_index = find(diff(W_ind) ~= 0, 1, 'last') + 1;
        num_morning_W = numel(stages) - first_morning_W_index + 1;
        
        nb_pre_post_sleep_wake_eps = num_evening_W + num_morning_W;
        if(nb_pre_post_sleep_wake_eps > second_largest)
            total_W_to_remove = nb_pre_post_sleep_wake_eps - second_largest;
            if(num_evening_W > total_W_to_remove)
                stages = stages(total_W_to_remove + 1 : end);
                eeg_epochs = eeg_epochs(total_W_to_remove + 1 : end, :);
                eog_epochs = eog_epochs(total_W_to_remove + 1 : end, :);
                emg_epochs = emg_epochs(total_W_to_remove + 1 : end, :);
            else
                evening_W_to_remove = num_evening_W;
                morning_W_to_remove = total_W_to_remove - evening_W_to_remove;
                stages = stages(evening_W_to_remove + 1 : end-morning_W_to_remove);
                eeg_epochs = eeg_epochs(evening_W_to_remove + 1 : end-morning_W_to_remove, :);
                eog_epochs = eog_epochs(evening_W_to_remove + 1 : end-morning_W_to_remove, :);
                emg_epochs = emg_epochs(evening_W_to_remove + 1 : end-morning_W_to_remove, :);
            end
        end
    else
        disp('Wake is not the biggest class, nothing to remove.')
    end
    
    % R&K to ASMM
    stages_from = [0, 1, 2, 3, 4, 5];
    stages_to = [1, 2, 3, 4, 4, 5];
    for i = numel(stages_from) : -1 : 1
        stages(stages == stages_from(i)) = stages_to(i);
    end
    y = zeros(numel(stages), 5);
    for i = 1 : numel(stages)
        y(i, stages(i)) = 1;
    end
    
    if(numel(unique(stages)) < 5)
        disp([filename, ': skipped less than 5 stages']) 
        ret = 0;
        return;
    end
    
    N = numel(stages);
    X_eeg = zeros(N, 29, nfft/2+1);
    for k = 1 : size(eeg_epochs, 1)
%         if(mod(k,100) == 0)
%             disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
%         end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X_eeg(k,:,:) = Xk;
    end
    
    X_eog = zeros(N, 29, nfft/2+1);
    for k = 1 : size(eog_epochs, 1)
%         if(mod(k,100) == 0)
%             disp([num2str(k),'/',num2str(size(eog_epochs, 1))]);
%         end
        [Xk,~,~] = spectrogram(eog_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X_eog(k,:,:) = Xk;
    end
    
    X_emg = zeros(N, 29, nfft/2+1);
    for k = 1 : size(emg_epochs, 1)
%         if(mod(k,100) == 0)
%             disp([num2str(k),'/',num2str(size(emg_epochs, 1))]);
%         end
        [Xk,~,~] = spectrogram(emg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X_emg(k,:,:) = Xk;
    end
    
    [N, t, f] = size(X_eeg);
    X_eeg_ = reshape(X_eeg,N,t*f);
    X_eog_ = reshape(X_eog,N,t*f);
    X_emg_ = reshape(X_emg,N,t*f);
    
    inf_ind = (isinf(sum(X_eeg_')) | isinf(sum(X_eog_')) | isinf(sum(X_emg_')));
    count = sum(inf_ind);
    clear X_eeg_ X_eog_ X_emg_
    
    if(count > 0)
        disp([num2str(n), ': ', num2str(count),' inf epochs removed']);
        stages(inf_ind) = [];
        y(inf_ind,:) = [];
        eeg_epochs(inf_ind,:) = [];
        X_eeg(inf_ind, :, :) = [];
        eog_epochs(inf_ind,:) = [];
        X_eog(inf_ind, :, :) = [];
        emg_epochs(inf_ind,:) = [];
        X_emg(inf_ind, :, :) = [];
    end
    
    assert(sum(isnan(X_eeg(:))) == 0, 'NaN');
    assert(sum(isnan(X_eog(:))) == 0, 'NaN');
    assert(sum(isnan(X_emg(:))) == 0, 'NaN');
    assert(sum(isinf(X_eeg(:))) == 0, 'Inf');
    assert(sum(isinf(X_eog(:))) == 0, 'Inf');
    assert(sum(isinf(X_emg(:))) == 0, 'Inf');
    
    % save data here
    y = single(y); % one-hot encoding
    label = single(stages);
    X2 = single(X_eeg);
    X1 = single(eeg_epochs);
    save([mat_path, 'n', num2str(n,'%04d'),'_eeg.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    disp([mat_path, 'n', num2str(n,'%04d'),'_eeg.mat'])
    
    X2 = single(X_eog);
    X1 = single(eog_epochs);
    save([mat_path, 'n', num2str(n,'%04d'),'_eog.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    disp([mat_path, 'n', num2str(n,'%04d'),'_eog.mat'])
    
    X2 = single(X_emg);
    X1 = single(emg_epochs);
    save([mat_path, 'n', num2str(n,'%04d'),'_emg.mat'], 'X1', 'X2', 'label', 'y', '-v7.3');
    disp([mat_path, 'n', num2str(n,'%04d'),'_emg.mat'])
    clear X1 X2 label y
end