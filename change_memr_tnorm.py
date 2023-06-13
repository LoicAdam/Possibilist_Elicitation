# -*- coding: utf-8 -*-

import os
import sys
import time
import multiprocessing
import itertools
from multiprocessing import Value
import numpy as np
from alternatives.data_preparation import generate_alternatives_score
from elicitation.models import ModelWeightedSum
from elicitation.elicitation import possibilist_elicitation
import matplotlib.pyplot as plt
import tikzplotlib
import pickle

nb_repetitions = 200 #Number of experiments
nb_alternatives = 30 #Number of alternatives
nb_parameters = 4 #Criteria                        
nb_questions = 5+1 #Nb iterations
path = 'results/change_memr_tnorm/'

def init_globals(counter):
    global cnt
    cnt = counter
    
def make_elicitation_tnorm(alternatives, model_values, confidence_values, rational,
                           t_norm):
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
    confidence_all = np.ones((nb_repetitions, nb_questions)) * 0.7
    model_values_all = np.random.dirichlet(np.ones(nb_parameters), size = nb_repetitions)
    rational_all = np.ones((nb_repetitions, nb_questions))
    

    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_min = pool.starmap(make_elicitation_tnorm, zip(alternatives_all, model_values_all,
                                                                   confidence_all, rational_all,
                                                                   np.repeat('minimum',nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time minimum: ", time.time() - start_time)
    memr_minimum = np.asarray([d['memr_list'] for d in elicitation_min if d is not None])

    ###

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_product = pool.starmap(make_elicitation_tnorm, zip(alternatives_all, model_values_all,
                                                                   confidence_all, rational_all,
                                                                   np.repeat('product',nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time product: ", time.time() - start_time)
    memr_product = np.asarray([d['memr_list'] for d in elicitation_product if d is not None])

    ###
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        elicitation_luka = pool.starmap(make_elicitation_tnorm, zip(alternatives_all, model_values_all,
                                                                   confidence_all, rational_all,
                                                                   np.repeat('lukasiewicz',nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time lukasiewicz: ", time.time() - start_time)
    memr_lukasiewicz = np.asarray([d['memr_list'] for d in elicitation_luka if d is not None])
    
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'res_' + str(confidence_all[0,0]) + '.pk','wb') as f:
        d = {}
        d['minimum'] = memr_minimum
        d['product'] = memr_product
        d['lukasiewicz'] = memr_lukasiewicz
        pickle.dump(d,f)
    
    memr_all = np.asarray([memr_minimum, memr_product, memr_lukasiewicz])
    memr_all_norm = memr_all / np.repeat(memr_all[:,:,0][:,:, np.newaxis], nb_questions, axis = 2)
    memr_all_norm = np.mean(memr_all_norm, axis = 1)

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,3)))
    fig, ax = plt.subplots(1,1)
    marker = itertools.cycle(('X', 'o', 'v')) 
    names = ['minimum', 'product', 'lukasiewicz']
    
    for con in range(0, 3):
        ax.plot(np.arange(0, nb_questions), memr_all_norm[con,:], next(marker),
                               linestyle='-', label=names[con])
    
    ax.set_xlabel("Number of questions")    
    ax.set_ylabel("mEMR/Initial mEMR")
    ax.xaxis.set_ticks(np.arange(0, 6, 1))
    ax.yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    ax.legend()
    plt.savefig(path + 'res_' + str(confidence_all[0,0]) + '.png', dpi=300)
    tikzplotlib.save(path + 'res_' + str(confidence_all[0,0]) + '.tex')
