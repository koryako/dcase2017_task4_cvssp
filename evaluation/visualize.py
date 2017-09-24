import numpy as np
import cPickle
import h5py
import time
import os
import gzip
import sys
import csv
from io_task4 import create_folder, get_my_open
from io_task4 import sed_read_prob_mat_list_to_csv
from io_task4 import sed_read_strong_gt_csv

def at_visualize(at_prob_mat_path, weak_gt_csv, lbs, out_path):
    lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
    create_folder(os.path.dirname(out_path))
    
    my_open = get_my_open(weak_gt_csv)
    with my_open(weak_gt_csv, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
    # Get gt_dict
    gt_dict = {}
    for li in lis:
        na = li[0]
        value = li[3]
        if na not in gt_dict.keys():
            gt_dict[na] = [value]
        else:
            gt_dict[na].append(value)
    
    with gzip.open(at_prob_mat_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
    
    # Create debug.csv
    f_write = open(out_path, 'w')
        
    for li in lis:
        na = li[0]
        values = li[1:]
        lbs = gt_dict[na]
        for lb in lbs:
            if lb in lb_to_idx.keys():
                index = lb_to_idx[lb]
                if not values[index][0] == '*':
                    values[index] = '*' + values[index]
            else:
                print lb, "not key (Please ignore)!"
        f_write.write(na)
        for i1 in xrange(len(values)):
            f_write.write('\t' + values[i1])
        f_write.write('\r\n')
        
    f_write.close()
    print "Write out at_visualize.csv successfully!\n"
    
    
def sed_visualize(sed_prob_mat_list_path, strong_gt_csv, lbs, step_sec, max_len):
    (pd_na_list, pd_prob_mat_list) = sed_read_prob_mat_list_to_csv(sed_prob_mat_list_path)
    (gt_na_list, gt_digit_mat_list) = sed_read_strong_gt_csv(
                                          strong_gt_csv=strong_gt_csv, 
                                          lbs=lbs, 
                                          step_sec=step_sec, 
                                          max_len=max_len)
    
    indexes = [pd_na_list.index(e) for e in gt_na_list]
    pd_prob_mat_list = [pd_prob_mat_list[idx] for idx in indexes]
    na_list = gt_na_list
    return na_list, pd_prob_mat_list, gt_digit_mat_list