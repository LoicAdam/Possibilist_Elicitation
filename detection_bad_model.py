# -*- coding: utf-8 -*-

import os
import sys
import time
import multiprocessing
import itertools
import pickle
from multiprocessing import Value
import numpy as np
from alternatives.data_preparation import generate_alternatives_score
from elicitation.models import ModelWeightedSum, ModelOWA
from elicitation.elicitation import possibilist_elicitation
import matplotlib.pyplot as plt
import tikzplotlib

t_norm = "product"
nb_repetitions = 200 #Number of experiments
nb_alternatives = 30 #Number of alternatives
nb_parameters = 4 #Criteria                        
nb_questions = 15+1 #Nb iterations
path = 'results/detection_bad_model/'

def init_globals(counter):
    global cnt
    cnt = counter
    
def make_elicitation_Random(alternatives, confidence_values):
    
    rational = np.random.randint(0, 2, nb_questions)
    model_values = np.random.dirichlet(np.ones(nb_parameters)/nb_parameters)
    res = possibilist_elicitation(alternatives, ModelWeightedSum(model_values),
                                  confidence_values, t_norm, max_iter = nb_questions,
                                  inconsistency_type = 'zero', rational = rational,
                                  stop_if_inconsistency = True)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res
    
def make_elicitation_OWA_Random(alternatives, confidence_values):
    
    rational = np.ones(nb_questions)
    model_values = np.random.dirichlet(np.ones(nb_parameters)/nb_parameters)
    res = possibilist_elicitation(alternatives, ModelWeightedSum(model_values),
                                  confidence_values, t_norm, max_iter = nb_questions,
                                  inconsistency_type = 'zero', rational = rational,
                                  stop_if_inconsistency = True, 
                                  true_model = ModelOWA(model_values))
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def make_elicitation_OWA_Fair(alternatives, confidence_values):
    
    rational = np.ones(nb_questions)
    parameters = np.zeros(nb_parameters)
    parameters[0] = 0.85
    parameters[1:nb_parameters] = np.repeat((1-0.85)/(nb_parameters-1),nb_parameters-1)
    model_values = np.random.dirichlet(parameters)
    res = possibilist_elicitation(alternatives, ModelWeightedSum(model_values),
                                  confidence_values, t_norm, max_iter = nb_questions,
                                  inconsistency_type = 'zero', rational = rational,
                                  stop_if_inconsistency = True, 
                                  true_model = ModelOWA(model_values))
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
    
    confidence_all =  np.round(np.random.beta(7, 2, size = ((nb_repetitions, nb_questions))),
                               decimals = 2)
            
    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)

    inconsistency_all = []
    func = [make_elicitation_Random, make_elicitation_OWA_Fair, make_elicitation_OWA_Random]
    
    for i in range(0,len(func)):
        start_time = time.time()
        cnt = Value('i', 0)
        with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
            elicitation = pool.starmap(func[i], zip(alternatives_all, confidence_all))
        sys.stdout.flush()
        pool.close()
        pool.join()
        real_list = [d['real_regret_list'] for d in elicitation if d is not None]
        inconsistency_list = [d['inconsistency_list'] for d in elicitation if d is not None]
        for j in range(0,len(real_list)):
            inconsistency = inconsistency_list[j]
            if len(inconsistency) != nb_questions:
                inconsistency_new = np.ones(nb_questions) * inconsistency[-1]
                inconsistency_new[0:len(inconsistency)] = inconsistency
                inconsistency_list[j] = inconsistency_new
        print("Time " + func[i].__name__ + ": ", time.time() - start_time)
        inconsistency_all.append(np.asarray(inconsistency_list))        
    
    inconsistency_all = np.asarray(inconsistency_all)
            
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'res.pk','wb') as f:
        d = {}
        d['i_Random'] = inconsistency_all[0]
        d['i_OWA_Random'] = inconsistency_all[1]
        d['i_OWA_Fair'] = inconsistency_all[2]
        pickle.dump(d,f)
        
    nb_detected = np.sum(inconsistency_all != 0, axis = 1)/nb_repetitions
    
    names = ["Random", "OWA Fair", "OWA Random"]
                
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,len(func))))
    fig, ax = plt.subplots(1,1)
    marker = itertools.cycle(('X', 'o', 'v')) 
    
    for con in range(0, len(func)):
        ax.plot(np.arange(0, nb_questions), nb_detected[con,:], next(marker),
                linestyle='-', label=names[con])
    
    ax.set_xlabel("Number of questions")    
    ax.set_ylabel("Percentage bad models detected")
    ax.xaxis.set_ticks(np.arange(0, nb_questions, 5))
    ax.yaxis.set_ticks(np.arange(0, 1.2, 0.2))
    ax.legend()
    plt.savefig(path + 'inc.png', dpi=300)
    tikzplotlib.save(path + 'inc.tex')
    