"""
SUMMARY:  Examples of evaluation code. 
AUTHOR:   Qiuqiang Kong, q.kong@surrey.ac.uk
Created:  2017.07.10
Modified: -
--------------------------------------
"""
import os
import evaluate
import io_task4
import visualize

lbs = ['Train horn', 'Air horn, truck horn', 'Car alarm', 'Reversing beeps', 
       'Bicycle', 'Skateboard', 'Ambulance (siren)', 
       'Fire engine, fire truck (siren)', 'Civil defense siren', 
       'Police car (siren)', 'Screaming', 'Car', 'Car passing by', 'Bus', 
       'Truck', 'Motorcycle', 'Train']

sample_rate = 16000.
n_window = 1024.
n_overlap = 360.     # ensure 240 frames / 10 seconds
max_len = 240        # sequence max length
step_sec = (n_window - n_overlap) / sample_rate     # step duration in second
       
strong_gt_csv = 'data/groundtruth_strong_label_testing_set.csv'
weak_gt_csv = 'data/groundtruth_weak_label_testing_set.csv'


### Audio tagging evaluations. 
def at_evaluate_from_prob_mat_csv():              
    at_prob_path = 'data/at_prob_mat.csv.gz'
    at_stat_path = '_stats/at_stat.csv'
    at_submission_path = '_submissions/at_submission.csv'
                       
    auto_thres = False
    if auto_thres:
        thres_ary = 'auto'
    else:
        thres_ary = [0.3] * len(lbs)
        
    at_evaluator = evaluate.AudioTaggingEvaluate(
                       weak_gt_csv=weak_gt_csv, 
                       lbs=lbs)
        
    stat = at_evaluator.get_stats_from_prob_mat_csv(
                 pd_prob_mat_csv=at_prob_path, 
                 thres_ary=thres_ary)
                 
    at_evaluator.write_stat_to_csv(stat=stat, 
                                     stat_path=at_stat_path)
                                     
    at_evaluator.print_stat(stat_path=at_stat_path)
    
    io_task4.at_write_prob_mat_csv_to_submission_csv(at_prob_mat_path=at_prob_path, 
                                                     lbs=lbs, 
                                                     thres_ary=stat['thres_ary'], 
                                                     out_path=at_submission_path)
                                    
def at_evaluate_from_submission_csv():
    at_submission_path = '_submissions/at_submission.csv'
    at_stat_path = '_stats/at_stat_from_submission_file.csv'
    
    at_evaluator = evaluate.AudioTaggingEvaluate(
                       weak_gt_csv=weak_gt_csv, 
                       lbs=lbs)
    
    stat = at_evaluator.get_stats_from_submit_format(
                 submission_csv=at_submission_path)
                 
    at_evaluator.write_stat_to_csv(
        stat=stat, 
        stat_path=at_stat_path)
        
    at_evaluator.print_stat(stat_path=at_stat_path)
    
def at_evaluate_ankit():
    at_submission_path = '_submissions/at_submission.csv'
    ankit_csv = 'evaluation_modified_ankitshah009/groundtruth/groundtruth_weak_label_testing_set.csv'
    at_stat_path = '_stats/at_stat_ankit.csv'
    
    at_evaluator = evaluate.AudioTaggingEvaluate(
                       weak_gt_csv=weak_gt_csv, 
                       lbs=lbs)
                       
    at_evaluator.write_out_ankit_stat(
        submission_csv=at_submission_path, 
        ankit_csv=ankit_csv, 
        stat_path=at_stat_path)
        
    at_evaluator.print_stat(at_stat_path)
    
### Sound event detection evaluations. 
def sed_evaluate_from_prob_mat_list_csv():
    sed_prob_mat_list_path = 'data/sed_prob_mat_list.csv.gz'
    sed_stat_path = '_stats/sed_stat.csv'
    sed_submission_path = '_submissions/sed_submission.csv'
    
    sed_evaluator = evaluate.SoundEventDetectionEvaluate(
                        strong_gt_csv=strong_gt_csv, 
                        lbs=lbs, 
                        step_sec=step_sec, 
                        max_len=max_len)
                        
    thres_ary = [0.3] * len(lbs)
    
    stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                 pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                 thres_ary=thres_ary)
                 
    sed_evaluator.write_stat_to_csv(stat=stat, 
                                      stat_path=sed_stat_path)
                                      
    sed_evaluator.print_stat(stat_path=sed_stat_path)
    
    io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
        sed_prob_mat_list_path=sed_prob_mat_list_path, 
        lbs=lbs, 
        thres_ary=thres_ary, 
        step_sec=step_sec, 
        out_path=sed_submission_path)
    
def sed_evaluate_from_submission_csv():
    sed_submission_path = '_submissions/sed_submission.csv'
    sed_stat_path = '_stats/sed_stat_from_submission.csv'
    
    sed_evaluator = evaluate.SoundEventDetectionEvaluate(
                        strong_gt_csv=strong_gt_csv, 
                        lbs=lbs, 
                        step_sec=step_sec, 
                        max_len=max_len)
                        
    stat = sed_evaluator.get_stats_from_submit_format(
                 submission_csv=sed_submission_path)
                 
    sed_evaluator.write_stat_to_csv(stat=stat, 
                                      stat_path=sed_stat_path)
                                      
    sed_evaluator.print_stat(stat_path=sed_stat_path)
    
### Visualizations. 
def at_visualize():
    at_prob_mat_path = 'data/at_prob_mat.csv.gz'
    out_path = '_visualizations/at_visualization.csv'
    
    visualize.at_visualize(at_prob_mat_path=at_prob_mat_path, 
                           weak_gt_csv=weak_gt_csv, 
                           lbs=lbs, 
                           out_path=out_path)
   
def sed_visualize():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    sed_prob_mat_list_path = 'data/sed_prob_mat_list.csv.gz'
    
    (na_list, pd_prob_mat_list, gt_digit_mat_list) = visualize.sed_visualize(
         sed_prob_mat_list_path=sed_prob_mat_list_path, 
         strong_gt_csv=strong_gt_csv, 
         lbs=lbs, 
         step_sec=step_sec, 
         max_len=max_len)
         
    for n in xrange(len(na_list)):
        na = na_list[n]
        pd_prob_mat = pd_prob_mat_list[n]
        gt_digit_mat = gt_digit_mat_list[n]
        
        fig, axs = plt.subplots(3, 1, sharex=True)
        
        axs[0].set_title(na + "\nYou may plot spectrogram here yourself. ")
        # axs[0].matshow(x.T, origin='lower', aspect='auto') # load & plot spectrogram here. 
        
        axs[1].set_title("Prediction")
        axs[1].matshow(pd_prob_mat.T, origin='lower', aspect='auto', vmin=0., vmax=1.)	
        axs[1].set_yticklabels([''] + lbs)
        axs[1].yaxis.set_major_locator(ticker.MultipleLocator(1))
        axs[1].yaxis.grid(color='w', linestyle='solid', linewidth=0.3)
        
        axs[2].set_title("Ground truth")
        axs[2].matshow(gt_digit_mat.T, origin='lower', aspect='auto', vmin=0., vmax=1.)	
        axs[2].set_yticklabels([''] + lbs)
        axs[2].yaxis.set_major_locator(ticker.MultipleLocator(1))
        axs[2].yaxis.grid(color='w', linestyle='solid', linewidth=0.3)
        plt.show()
        
    
### main
if __name__ == '__main__':
    if not os.path.exists('_submissions'): os.makedirs('_submissions')
    if not os.path.exists('_stats'): os.makedirs('_stats')
    
    at_type = 0
    sed_type = 0
    ankit_evaluate = False
    at_vis = True
    sed_vis = True
    
    print "============= Audio tagging Evaluation ============="
    if at_type == 0:
        at_evaluate_from_prob_mat_csv()
    elif at_type == 1:
        at_evaluate_from_submission_csv()
        
    if ankit_evaluate:
        at_evaluate_ankit()
    
    print "============= Sound event detection Evaluation ============="
    if sed_type == 0:
        sed_evaluate_from_prob_mat_list_csv()
    elif sed_type == 1:
        sed_evaluate_from_submission_csv()
        
    print "============= Visualizations ============="
    if at_vis:
        at_visualize()
        
    if sed_vis:
        sed_visualize()