1. To prepare data
run shhs_data.m

2. Split data and generate file lists
run data_split_eval.m
run genlist_scratch_training.m
(Not: I have included the "data_split_eval.mat" file and the "file_list" folder, you dont have to run this step again)

3. Train and evaluate the networks
run bash scripts in "scratch_training/sleeptransformer". The environment I used was Tensorflow 1.13, Python 3.6

4. Run matlab scripts in "evaluation" folders to aggregate the network outputs and compute metrics
for example, run aggregate_sleeptransformer.m


