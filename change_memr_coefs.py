'''Evolution of the mEMR depending on the confidence degree for a fixed T-norm'''

import os
import sys
import time
import multiprocessing
import itertools
import pickle
from multiprocessing import Value
import numpy as np
from alternatives.data_preparation import generate_alternatives_score
from elicitation.models import ModelWeightedSum
from elicitation.elicitation import possibilist_elicitation
import matplotlib.pyplot as plt
import tikzplotlib

nb_repetitions = 200 #Number of experiments
nb_alternatives = 30 #Number of alternatives
nb_parameters = 4 #Criteria                        
nb_questions = 5+1 #Nb iterations
confs = [0,0.3,0.5,0.7,0.8,0.9,1]
path = 'results/change_memr_coefs/'

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
    model_values_all = np.random.dirichlet(np.ones(nb_parameters), size = nb_repetitions)
    rational_all = np.ones((nb_repetitions, nb_questions))

    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)
    
    res_all = []
    for i in range(0,len(confs)):
        confidence_all = np.ones((nb_repetitions, nb_questions)) * confs[i]
        start_time = time.time()
        cnt = Value('i', 0)
        with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
            elicitation_minimum = pool.starmap(make_elicitation_tnorm, zip(alternatives_all, model_values_all,
                                                                           confidence_all, rational_all,
                                                                           np.repeat('minimum',nb_repetitions)))
        sys.stdout.flush()
        pool.close()
        pool.join()
        memr_list = [d['memr_list'] for d in elicitation_minimum if d is not None]
        for j in range(0,len(memr_list)):
            memr = memr_list[j]
            if len(memr) != nb_questions:
                memr_new = np.ones(nb_questions) * memr[-1]
                memr_new[0:len(memr)] = memr
                memr_list[j] = memr_new
        
        print("Time minimum " + str(confs[i]) + ": ", time.time() - start_time)
        res_all.append(np.asarray(memr_list))
    memr_minimum = np.asarray(res_all)
        
    ###

    res_all = []
    for i in range(0,len(confs)):
        confidence_all = np.ones((nb_repetitions, nb_questions)) * confs[i]
        start_time = time.time()
        cnt = Value('i', 0)
        with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
            elicitation_product = pool.starmap(make_elicitation_tnorm, zip(alternatives_all, model_values_all,
                                                                           confidence_all, rational_all,
                                                                           np.repeat('product',nb_repetitions)))
        sys.stdout.flush()
        pool.close()
        pool.join()
        memr_list = [d['memr_list'] for d in elicitation_product if d is not None]
        for j in range(0,len(memr_list)):
            memr = memr_list[j]
            if len(memr) != nb_questions:
                memr_new = np.ones(nb_questions) * memr[-1]
                memr_new[0:len(memr)] = memr
                memr_list[j] = memr_new
        
        print("Time product " + str(confs[i]) + ": ", time.time() - start_time)
        res_all.append(np.asarray(memr_list))
    memr_product = np.asarray(res_all)
            
    ###
        
    res_all = []
    for i in range(0,len(confs)):
        confidence_all = np.ones((nb_repetitions, nb_questions)) * confs[i]
        start_time = time.time()
        cnt = Value('i', 0)
        with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
            elicitation_luka = pool.starmap(make_elicitation_tnorm, zip(alternatives_all, model_values_all,
                                                                        confidence_all, rational_all,
                                                                        np.repeat('lukasiewicz',nb_repetitions)))
        sys.stdout.flush()
        pool.close()
        pool.join()
        memr_list = [d['memr_list'] for d in elicitation_luka if d is not None]
        for j in range(0,len(memr_list)):
            memr = memr_list[j]
            if len(memr) != nb_questions:
                memr_new = np.ones(nb_questions) * memr[-1]
                memr_new[0:len(memr)] = memr
                memr_list[j] = memr_new
        
        print("Time luka " + str(confs[i]) + ": ", time.time() - start_time)
        res_all.append(np.asarray(memr_list))
    memr_luka = np.asarray(res_all)
    
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'res.pk','wb') as f:
        d = {}
        d['minimum'] = memr_minimum
        d['product'] = memr_product
        d['lukasiewicz'] = memr_luka
        pickle.dump(d,f)
        
    memr_all = np.asarray([memr_minimum, memr_product, memr_luka])
    memr_all_norm = memr_all / np.repeat(memr_all[:,:,:,0][:,:,:,np.newaxis], nb_questions, axis = 3)
    memr_all_norm = np.mean(memr_all_norm, axis = 2)
    names = ['minimum', 'product', 'lukasiewicz']

    for i in range(0,3):
        
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,len(confs))))
        fig, ax = plt.subplots(1,1)
        marker = itertools.cycle(('X', 'o', 'v', 's', 'p', '*', 'D')) 
        
        for con in range(0, len(confs)):
            ax.plot(np.arange(0, nb_questions), memr_all_norm[i,con,:], next(marker),
                    linestyle='-', label=confs[con])
        
        ax.set_xlabel("Number of questions")    
        ax.set_ylabel("mEMR/Initial mEMR")
        ax.xaxis.set_ticks(np.arange(0, 6, 1))
        ax.yaxis.set_ticks(np.arange(0, 1.2, 0.2))
        ax.legend(confs)
        plt.savefig(path + 'res_' + str(names[i]) + '.png', dpi=300)
        tikzplotlib.save(path + 'res_' + str(names[i]) + '.tex')
        