# -*- coding: utf-8 -*-

import os
import sys
import time
import multiprocessing
import itertools
import pickle
import copy
from multiprocessing import Value
import numpy as np
from alternatives.data_preparation import generate_alternatives_score
from elicitation.models import ModelWeightedSum
from elicitation.elicitation import possibilist_elicitation
import matplotlib.pyplot as plt
import tikzplotlib
      
t_norm = "product"
nb_repetitions = 20 #Number of experiments
nb_alternatives = 10 #Number of alternatives
nb_parameters = 3 #Criteria                        
nb_questions = 5+1 #Nb iterations
confs = [0.5, 0.7, 0.9, 1]
path = 'results/change_inconsistency/'

def init_globals(counter):
    global cnt
    cnt = counter
    
def make_elicitation_inconsistency(alternatives, model_values, confidence_values,
                                   rational):
    res = possibilist_elicitation(alternatives, ModelWeightedSum(model_values),
                                  confidence_values, t_norm, max_iter = nb_questions,
                                  inconsistency_type = 'zero', rational = rational)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

if __name__ == '__main__':

    alternatives_all = np.zeros((nb_repetitions, nb_alternatives, nb_parameters))
    for i in range(0, nb_repetitions):
        alternatives_all[i,:,:] = generate_alternatives_score(nb_alternatives,
                                                              nb_parameters = nb_parameters,
                                                              value = nb_parameters/2)
    model_values_all = np.random.dirichlet(np.ones(nb_parameters), size = nb_repetitions)
    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)
    
    random_mask = np.random.uniform(size = (nb_repetitions, nb_questions))
    fake_confidence = np.ones((nb_repetitions, nb_questions)) * 0.7
    rational_all = np.where(random_mask <= fake_confidence + (1-fake_confidence)/2, 1, 0)
    non_zeros_lines = np.count_nonzero(rational_all, axis = 1) #We add an error if none.
    for j in range(0, nb_repetitions):
        if non_zeros_lines[j] == nb_questions:
            rational_all[j, np.random.randint(0, nb_questions)] = 0
            
    real_all = []
    inconsistency_all = []
    for i in range(0,len(confs)):
        confidence_all = np.ones((nb_repetitions, nb_questions)) * confs[i]
        start_time = time.time()
        cnt = Value('i', 0)
        with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
            elicitation = pool.starmap(make_elicitation_inconsistency, zip(alternatives_all, model_values_all,
                                                                           confidence_all, rational_all))
        sys.stdout.flush()
        pool.close()
        pool.join()
        real_list = [d['real_regret_list'] for d in elicitation if d is not None]
        inconsistency_list = [d['inconsistency_list'] for d in elicitation if d is not None]
        for j in range(0,len(real_list)):
            real = real_list[j]
            inconsistency = inconsistency_list[j]
            if len(real) != nb_questions:
                real_new = np.ones(nb_questions) * real[-1]
                real_new[0:len(real)] = real
                real_list[j] = real_new
                inconsistency_new = np.ones(nb_questions) * inconsistency[-1]
                inconsistency_new[0:len(inconsistency)] = inconsistency
                inconsistency_list[j] = inconsistency_new
        print("Time " + str(confs[i]) + ": ", time.time() - start_time)
        real_all.append(np.asarray(real_list))
        inconsistency_all.append(np.asarray(inconsistency_list))        
    
    real_all = np.asarray(real_all)
    real_stop_when_inconsistency_detected_all = copy.deepcopy(real_all)
    inconsistency_all = np.asarray(inconsistency_all)
    
    for i in range(0, real_all.shape[0]):
        for j in range(0, real_all.shape[1]):
            inconsistency_detected = np.nonzero(inconsistency_all[i,j,:])[0]
            if len(inconsistency_detected) != 0:
                real_before = real_all[i,j,inconsistency_detected[0]]
                real_stop_when_inconsistency_detected_all[i,j,inconsistency_detected] = np.repeat(real_before, len(inconsistency_detected))
            
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'res_' + t_norm + '.pk','wb') as f:
        d = {}
        d['real'] = real_all
        d['real_detected'] = real_stop_when_inconsistency_detected_all
        d['inconsistency'] = inconsistency_all
        pickle.dump(d,f)
        
    inconsistency_norm = np.mean(inconsistency_all, axis = 1)
                
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,len(confs))))
    fig, ax = plt.subplots(1,1)
    marker = itertools.cycle(('X', 'o', 'v', 's')) 
    
    for con in range(0, len(confs)):
        ax.plot(np.arange(0, nb_questions), inconsistency_norm[con,:], next(marker),
                linestyle='-', label=confs[con])
    
    ax.set_xlabel("Number of questions")    
    ax.set_ylabel("Inconsistency")
    ax.xaxis.set_ticks(np.arange(0, nb_questions, 5))
    ax.yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    ax.legend(confs)
    plt.savefig(path + 'inc_' + t_norm + '.png', dpi=300)
    tikzplotlib.save(path + 'inc_' + t_norm + '.tex')
    
    ###
    
    real_stop_when_inconsistency_detected_all_norm = np.mean(real_stop_when_inconsistency_detected_all, axis = 1)
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,len(confs))))
    fig, ax = plt.subplots(1,1)
    marker = itertools.cycle(('X', 'o', 'v', 's')) 
    
    for con in range(0, len(confs)):
        ax.plot(np.arange(0, nb_questions), real_stop_when_inconsistency_detected_all_norm[con,:], next(marker),
                linestyle='-', label=confs[con])
    
    ax.set_xlabel("Number of questions")    
    ax.set_ylabel("mEMR/Initial mEMR")
    ax.xaxis.set_ticks(np.arange(0, nb_questions, 5))
    ax.legend(confs)
    plt.savefig(path + 'real_' + t_norm + '.png', dpi=300)
    tikzplotlib.save(path + 'real_' + t_norm + '.tex')
        