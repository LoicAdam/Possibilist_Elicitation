# -*- coding: utf-8 -*-
"""Elicitation"""

import time
import numpy as np
from alternatives.data_preparation import get_pareto_efficient_alternatives
from elicitation.question_strategies import CSSQuestionStrategy
from elicitation.dm import get_choice_fixed
from elicitation.choice_calculation import pmr_polytope, mr_polytope
from elicitation.focal_set import compute_epmr_emr
from elicitation.choice_strategies import minimax_regret_choice
from elicitation.polytope import Polytope, construct_constrainst, cut_polytope, intersection_checker

def possibilist_elicitation(alternatives, model, confidence, t_norm = 'product',
                            max_iter = -1, inconsistency_type = 'zero',
                            rational = None, true_model = None,
                            stop_if_inconsistency = False,
                            regret_limit = 10**-10, min_possibility = 0):
    """
    Possibilist elicitation with CSS.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    model : Model
        Model.
    confidence : list
        Confidence degrees.
    t_norm : string, optional
        Which T-norm to use. The default is 'product'.
    max_iter : integer, optional
        If a maximum interation and how much. The default is -1.
    inconsistency_type : string, optional
        Inconsistency in the EPMR. The default is 'zero'.
    rational : list, optional
        To know if some answers should be rational or not. The default is None.
    true_model : Model, Optional
        Decision strategy for the decision maker. Optional if model already is the true model. The default is None.
    stop_if_inconsistency : bool, Optional
        Should we stop if inconsistency? The default is False.
    regret_limit : float, optional
        If a regret limit. The default is 10**-10.
    min_possibility : float, optional
        Min possibility to consider a polytope. The default is 0.

    Returns
    -------
    dict
        Elicitation information.

    """
    alternatives = get_pareto_efficient_alternatives(alternatives) #Get rid of non optimal solutions.
    nb_alternatives = len(alternatives)

    constraints = model.get_model_constrainsts()
    constraints_a = constraints['A_eq']
    constraints_b = constraints['b_eq']
    bounds = constraints['bounds']
    first_polytope = Polytope(None,None,constraints_a, constraints_b, bounds)

    polytope_list = []
    polytope_list.append(first_polytope)

    question_strategy = CSSQuestionStrategy(alternatives)
    
    if true_model == None:
        true_model = model

    #Maximal number of iterations.
    number_pairs = int(nb_alternatives*(nb_alternatives-1)/2)
    if max_iter != -1:
        max_iter = np.minimum(max_iter, number_pairs)
    else:
        max_iter = number_pairs
        
    memr_estimated_list = np.zeros(max_iter)
    mmr_real_list = np.zeros(max_iter)
    inconsistency_list = np.zeros(max_iter)

    scores = true_model.get_model_score(alternatives)

    if rational is None:
        rational = np.ones(max_iter)

    A_list = []
    b_list = []

    ite = 0
    start_time = time.time()

    epmr = pmr_polytope(alternatives, first_polytope, model)
    emr =  mr_polytope(epmr)

    while ite < max_iter:

        pmr_list = []
        possibility_list = []
        new_polytope_list = []

        candidate_alt, candidate_alt_id = question_strategy.give_candidate(emr)
        worst_alt, _ = question_strategy.give_oponent(epmr, candidate_alt_id)
        choice = get_choice_fixed(candidate_alt, worst_alt, rational[ite], true_model)
        best_prefered = choice['accepted']
        _, best_alt_id, regret = minimax_regret_choice(alternatives, emr)
        new_constraint_a, new_constraint_b = construct_constrainst(candidate_alt, worst_alt, best_prefered, model)
        A_list.append(new_constraint_a)
        b_list.append(new_constraint_b)

        for polytope in polytope_list:

            side = intersection_checker(polytope, new_constraint_a, new_constraint_b)
            if side is None:
                return None
            '''If the new constrainst intersects with the current polytope:
            - Create two new ones,
            - Keep those with a suffissant possibility,
            - Delete the original polytope'''
            if side == 0:
                polytope_1, polytope_2 = cut_polytope(polytope, new_constraint_a, new_constraint_b, confidence[ite], t_norm)
                del polytope
                if polytope_1.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_1)
                    pmr = pmr_polytope(alternatives, polytope_1, model)
                    pmr_list.append(pmr)
                    possibility_list.append(polytope_1.get_possibility())
                else:
                    del polytope_1
                if polytope_2.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_2)
                    pmr = pmr_polytope(alternatives, polytope_2, model)
                    pmr_list.append(pmr)
                    possibility_list.append(polytope_2.get_possibility())

                else:
                    del polytope_2

            #Else, just update the possibility.
            else:
                if side == 1:
                    polytope.add_answer(new_constraint_a, new_constraint_b, 1, t_norm)
                else:
                    polytope.add_answer(-new_constraint_a, -new_constraint_b, 1-confidence[ite], t_norm)

                if polytope.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope)
                    pmr = pmr_polytope(alternatives, polytope, model)
                    pmr_list.append(pmr)
                    possibility_list.append(polytope.get_possibility())
                else:
                    del polytope

        polytope_list = new_polytope_list
        
        if len(possibility_list) == 0:
            inconsistency_list[ite] = inconsistency_list[ite-1]
        else:
            inconsistency_list[ite] = 1-np.max(possibility_list)
        mmr_real_list[ite] = np.max(scores) - scores[best_alt_id]
        memr_estimated_list[ite] = np.max(regret)
        
        if (np.max(regret) <= regret_limit) or (stop_if_inconsistency and inconsistency_list[ite] > 0):
            break
        
        epmr, emr = compute_epmr_emr(pmr_list, possibility_list, inconsistency_type)
        ite = ite+1

    memr_estimated_list = memr_estimated_list[0:ite+1]
    mmr_real_list = mmr_real_list[0:ite+1]
    inconsistency_list = inconsistency_list[0:ite+1]
    d = {}
    d['time'] = time.time() - start_time
    d['memr_list'] = memr_estimated_list
    d['real_regret_list'] = mmr_real_list
    d['inconsistency_list'] = inconsistency_list
    d['possibility_list'] = possibility_list
    d['polytope_list'] = polytope_list
    d['best_alternative'] = best_alt_id
    d['value_list'] = pmr_list
    d['A'] = np.concatenate(A_list, axis=0)
    d['b'] = np.concatenate(b_list, axis=0)
    d['ite'] = ite
    return d

def get_recommendation(things_list, possibility_list, alternatives, model,
                       inconsistency_type = 'zero', polytopes = True):
    """
    Determine the optimal recommendation according to some criterion from polytopes or values.

    Parameters
    ----------
    things_list : list
        List of polytopes or values.
    possibility_list : list
        List of possibility for each polytope.
    alternatives : array_like
        Alternatives.
    model : Model
        The model.
    inconsistency_type : string, optional
        Inconsistency in the EPMR. The default is 'zero'.
    polytopes : bool, optional
        Do we use polytopes in things_list. The default is True.
        
    Returns
    -------
    dict
        Information about the recommended alternative.

    """
    scores = model.get_model_score(alternatives)
    if polytopes is True:
        value_list = []
        for polytope in things_list:
            value_list.append(pmr_polytope(alternatives, polytope, model))
    else:
        value_list = things_list

    _, epmr = compute_epmr_emr(value_list, possibility_list, inconsistency_type)
    _, best_alt_id, _ = minimax_regret_choice(alternatives, epmr)
    regret_real = np.max(scores) - scores[best_alt_id]

    result = {}
    result['best_alternative'] = best_alt_id
    result['real_regret'] = regret_real
    if polytopes is True:
        result['value_list'] = value_list
    return result
