% This script is to normalize and export spectral features into a format
% which is ready to train a CNN on Tensorflow

clear all
close all
clc

% this is exactly the same as for rnn

rng(10); % for repeatable

mat_path = './mat/';
load('./data_split_eval.mat');

tf_path = './file_list/scratch_training/eeg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

train_filename = [tf_path, 'train_list.txt'];
fid = fopen(train_filename,'wt');
for i = 1 : numel(train_sub)
    sname = ['n', num2str(train_sub(i),'%04d'),'_eeg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

train_filename = [tf_path, 'train_list_check.txt'];
fid = fopen(train_filename,'wt');
for i = 1 : numel(train_check_sub)
    sname = ['n', num2str(train_check_sub(i),'%04d'),'_eeg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

eval_filename = [tf_path, 'eval_list.txt'];
fid = fopen(eval_filename,'wt');
for i = 1 : numel(eval_sub)
    sname = ['n', num2str(eval_sub(i),'%04d'),'_eeg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

test_filename = [tf_path, 'test_list.txt'];
fid = fopen(test_filename,'wt');
for i = 1 : numel(test_sub)
    sname = ['n', num2str(test_sub(i),'%04d'),'_eeg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path


tf_path = './file_list/scratch_training/eog/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

train_filename = [tf_path, 'train_list.txt'];
fid = fopen(train_filename,'wt');
for i = 1 : numel(train_sub)
    sname = ['n', num2str(train_sub(i),'%04d'),'_eog.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

train_filename = [tf_path, 'train_list_check.txt'];
fid = fopen(train_filename,'wt');
for i = 1 : numel(train_check_sub)
    sname = ['n', num2str(train_check_sub(i),'%04d'),'_eog.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

eval_filename = [tf_path, 'eval_list.txt'];
fid = fopen(eval_filename,'wt');
for i = 1 : numel(eval_sub)
    sname = ['n', num2str(eval_sub(i),'%04d'),'_eog.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

test_filename = [tf_path, 'test_list.txt'];
fid = fopen(test_filename,'wt');
for i = 1 : numel(test_sub)
    sname = ['n', num2str(test_sub(i),'%04d'),'_eog.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path


tf_path = './file_list/scratch_training/emg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

train_filename = [tf_path, 'train_list.txt'];
fid = fopen(train_filename,'wt');
for i = 1 : numel(train_sub)
    sname = ['n', num2str(train_sub(i),'%04d'),'_emg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

train_filename = [tf_path, 'train_list_check.txt'];
fid = fopen(train_filename,'wt');
for i = 1 : numel(train_check_sub)
    sname = ['n', num2str(train_check_sub(i),'%04d'),'_emg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

eval_filename = [tf_path, 'eval_list.txt'];
fid = fopen(eval_filename,'wt');
for i = 1 : numel(eval_sub)
    sname = ['n', num2str(eval_sub(i),'%04d'),'_emg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path

test_filename = [tf_path, 'test_list.txt'];
fid = fopen(test_filename,'wt');
for i = 1 : numel(test_sub)
    sname = ['n', num2str(test_sub(i),'%04d'),'_emg.mat'];
    load([mat_path,sname], 'label');
    num_sample = numel(label);
    file_path = ['../../mat/',sname];
    fprintf(fid, '%s\t%d\n', file_path, num_sample);
end
fclose(fid);
clear fid file_path



