# -*- coding: utf-8 -*-
"""
Created on Thu May 20 09:15:25 2021

@author: sarab
"""


from collections import Counter
from typing import List, Tuple

import globals
import log
import pandas as pd

VUL_PERCENTAGE_TARGET = .374
NV_PERCENTAGE_TARGET = .455
MAX_ITERATIONS = 500
STARTING_INCREMENT = 1

import numpy as np
def calculatePointCount(features : np.ndarray, vectors : List[List[int]]) -> np.ndarray:
    # Features is an N x M matrix with N deals and M features per deal.
    # Vectors is a list of V vectors, each with M weights.
    # The output of this method is an N x V  matrix containing the points for
    #   each deal as evaluated by each vector
    #
    # To get the point counts, we must convert 
    #   the vectors into an M x V matrix. We then multiply this matrix
    #   with the features matrix to get an N x V matrix of results.    
    return np.dot(features, np.column_stack(vectors))

def calculateAccuracy(pointCounts : np.ndarray, thresholds : np.ndarray, targets : np.ndarray) -> List[int]:
    # PointCounts is an N x V matrix containing the point count for each of N deals
    #   as evaulated by each of V vectors
    #   The matrix can be obtained by calling calculatePointCount.
    # Thresholds is an N x V matrix of thresholds for N deals and V vectors  
    #   It represents the point count for which we predict 3NT is makable.
    # Targets is an N x 1 matrix indication whether 3NT makes for the
    #   corresponding deal.
    #
    # The output of this method is a V-element list containing the percentage of
    #   accurate predictions for each vector
    #
    # First we form an N x V matrix containing True
    #   if the vth vector predicts 3NT makes for the nth deal. We take the
    #   exclusive or of this matrix with the target matrix, yielding True
    #   if the vector's prediction is incorrect and False if it is correct.
    #   This is backwards from what we really want, but it's convenient, since
    #   we then calculate the average number of Falses to get the percentage 
    #   of correct predictions.
    
    results = np.logical_xor(pointCounts >= thresholds, targets)
    return (1 - sum(results) / len(pointCounts)).tolist()

def calculateExpectation(pointCounts : np.ndarray, thresholds : np.ndarray, targets : np.ndarray) -> List[int]:
    # PointCounts is an N x V matrix containing the point count for each of N deals
    #   as evaulated by each of V vectors
    #   The matrix can be obtained by calling calculatePointCount.
    # Thresholds is an N x V matrix of thresholds for N deals and V vectors  
    #   It represents the point count for which we predict 3NT is makable.
    # Targets is an N x 1 matrix indicating our score if we bid
    #   3NT (the negative is our score if we don't bid 3NT)
    #
    # The output of this method is a V-element list containing the score for each vector
    #
    #   of correct predictions.
    
    bidGame = pointCounts >= thresholds
    return (sum(bidGame * targets - ~bidGame * targets) / globals.deals.number_of_deals).tolist()

def calculatePayoff(pointCounts : np.ndarray, thresholds : np.ndarray, targets : np.ndarray) -> List[int]:
     if globals.payoff_metric == 'expectation':
         return calculateExpectation(pointCounts, thresholds, targets)
     elif globals.payoff_metric == 'accuracy':
         return calculateAccuracy(pointCounts, thresholds, targets)
     else:
         assert False, f'Unsupported metric {globals.payoff_metric}'
  

def calculateImpResults(pointCounts : np.ndarray, thresholds : np.ndarray, scores : np.ndarray) -> np.ndarray:
   # PointCounts is an N x V matrix containing the point count for each of N deals
    #   as evaulated by each of V vectors
    #   The matrix can be obtained by calling calculatePointCount.
     # Thresholds is an N x V matrix of thresholds for N deals and V vectors  
    #   It represents the point count for which we predict 3NT is makable.
    # Scores is an N X 1 matrix containing the imp score you receive if you play
    #   3NT and the other table plays 2NT
    # This method returns a recap sheet, that is, an N x V matrix where
    #   each entry is the score on the Nth board for the Vth vector
    #   The score is calculated by assuming each vector bids game if its
    #   point count exceed the threshold for the veulnerability of the 
    #   given board. Its result is then imped against every other vector
    #
    # To get the score, we construct a matrix indicating whether we bid game
    #   by comparing pointcount with threshold. Then we apply a lambda function 
    #   to each row. If the entry is True, the 
    #   lambda function returns the number of other tables that fails to bid 3NT.
    #   If the entry is False, the lambda function returns minus the number of
    #   other tables that bid 3NT. If we multiple the resulting matrix with
    #   the 'score' for that board, we receive our imp score.
    return  scores * np.apply_along_axis(lambda row: 
        [len(row) * x  - sum(row) for x in row], 1, 
        pointCounts >= thresholds)
 
def reportMatchResults(recapSheet : np.ndarray):
    relevantBoards = recapSheet[np.prod(recapSheet, axis = 1) != 0]
    return len(relevantBoards) / len(recapSheet), (np.average(relevantBoards, axis = 0) / (recapSheet.shape[1] - 1)).tolist()
       
def runSimulatedMatch(pointCounts : np.ndarray, thresholds : np.ndarray, scores : np.ndarray) -> List[int]:
    # PointCounts is an N x V matrix containing the point count for each of N deals
    #   as evaulated by each of V vectors
    #   The matrix can be obtained by calling calculatePointCount.
    # Thresholds is an N x V matrix of thresholds for N deals and V vectors  
    #   It represents the point count for which we predict 3NT is makable.
    # Scores is an N X 1 matrix containing the imp score you receive if you play
    #   3NT and the other table plays 2NT
    #
    # The output of this method is a tuple consisting of
    #   1) the percentage of boards on which someone did something different
    #   2) a V-element list containing the average score per board each vector receives 
    #     if imped against each of the other vectors
    
    return reportMatchResults(calculateImpResults(pointCounts, thresholds, scores))

# not current used but may wish to incorporate later
def analyzeMatchResults(recapSheet : np.ndarray):
    categories = [int(x * 2 + y) for x, y in zip(globals.deals.getVulnerabilities(), globals.deals.getTargets())]   
    return pd.DataFrame( {'good_game_nv' :
            sum((recapSheet > 0) * np.array([x == 1 for x in categories]).reshape(globals.deals.number_of_deals, 1)),
        'good_stop_nv':
            sum((recapSheet > 0) * np.array([x == 0 for x in categories]).reshape(globals.deals.number_of_deals, 1)),
        'good_game_vul':
            sum((recapSheet > 0) * np.array([x == 3 for x in categories]).reshape(globals.deals.number_of_deals, 1)),
        'good_stop_vul':
            sum((recapSheet > 0) * np.array([x == 2 for x in categories]).reshape(globals.deals.number_of_deals, 1)),
        'underbid_nv':
            sum((recapSheet < 0) * np.array([x == 1 for x in categories]).reshape(globals.deals.number_of_deals, 1)),
        'overbid_nv':
            sum((recapSheet < 0) * np.array([x == 0 for x in categories]).reshape(globals.deals.number_of_deals, 1)),
        'underbid_vul':
            sum((recapSheet < 0) * np.array([x == 3 for x in categories]).reshape(globals.deals.number_of_deals, 1)),
        'overbid_vul':
            sum((recapSheet < 0) * np.array([x == 2 for x in categories]).reshape(globals.deals.number_of_deals, 1))},
        index = globals.vectors.getVectorNames())
           
def calculateSuccessesByPointCount(targets : np.ndarray, pointCounts : np.ndarray) -> pd.core.frame.DataFrame:
    # Targets an n-element List specifying whether each deal makes 3NT
    # PointCounts is a matrix of the same dimension, containing the point count for each deal.
    #
    # The output of this method is a Series, indexed by point count.
    #   The values contains the percentage of times 3NT makes for deals 
    #     with that point count.
    outcomes = pd.concat([pd.DataFrame.from_dict(Counter(pointCounts.flatten() * targets.flatten()), orient='index', columns=['makes']),
        pd.DataFrame.from_dict(Counter(pointCounts.flatten()), orient='index', columns=['totals'])],
        axis = 1).fillna(0).drop(0).sort_index(ascending=False)
    # drop rows where we have few samples
    outcomes.drop(outcomes[outcomes.totals <  .2 * globals.deals.number_of_deals / len(outcomes)].index, inplace = True)
    return pd.Series( outcomes.apply(lambda row: row.makes / row.totals, axis = 1))


def calculatePointCountForTargetSuccessRate(successes : pd.Series, targetSuccessRate : float) -> int:
    # Successes is a Series, indexed by point count.
    #   The values contains the percentage of times 3NT makes for deals 
    #     with that point count.
    #   This Series can be constructed by using the calculateSuccessesByPointCount method.
    # TargetSuccessRate is the probability we need a game to succeed in order to bid it.
    #
    # The output of this method is the minimum point count we should choose to bid a game
    #   that has the targetSuccessRate chance of making.
    return min(successes[successes >= targetSuccessRate].index)
    
def calculateThresholdsByVulnerability(targets : np.ndarray, pointCounts : np.ndarray) -> Tuple[int, int]:
    # Targets an n-element List specifying whether each deal makes 3NT
    # PointCounts is a matrix of the same dimension, containing the point count for each deal.
    #
    # The output of this method is a tuple, consisting of the minimum point counts we
    #   need to bid a vul game and a non_vul game
    successes = calculateSuccessesByPointCount(targets, pointCounts)
    return (calculatePointCountForTargetSuccessRate(successes, VUL_PERCENTAGE_TARGET),
            calculatePointCountForTargetSuccessRate(successes, NV_PERCENTAGE_TARGET))

def calculateAllThresholdsByVulnerability(targets : np.ndarray, pointCounts : np.ndarray) -> np.ndarray:
    # Targets an n-element List specifying whether each deal makes 3NT
    # PointCounts is a V x N matrix containing the point count for each deal and each vector
    #
    # The output of this method is a V x 2 matrix, consisting of the minimum point counts we
    #   need to bid a vul game and a non_vul game for each vector
    thresholds = np.empty
    for v in range(pointCounts.shape[1]):
        if thresholds is np.empty:
            thresholds = calculateThresholdsByVulnerability(targets, pointCounts[:,v])
        else:
            thresholds = np.vstack((thresholds, calculateThresholdsByVulnerability(targets, pointCounts[:,v])))
    return thresholds

def bumpUp(vector : List[int], ordered_set : List[int], index : int, minValue : int) -> List[int]:
    # ordered_set[index] (if index is valid) must be at least minValue
    # if it returns an empty vector, the adjustment could not be made because of contraints
    new_vector = vector.copy()
    if index < len(ordered_set):
        feature = ordered_set[index]
        if new_vector[feature] < minValue:
            if feature in globals.features.getFixedFeatures():
                new_vector = []
            else:
                new_vector = adjustFeature(new_vector, feature, minValue - new_vector[feature])
    return new_vector

def bumpDown(vector : List[int], ordered_set : List[int], index : int, maxValue : int) -> List[int]:
    # ordered_set[index] (if index is valid) must be at most maxValue
    # if it returns an empty vector, the adjustment could not be made because of contraints
    new_vector = vector.copy()
    if index >= 0:
        feature = ordered_set[index]
        if new_vector[feature]> maxValue:
            if feature in globals.features.getFixedFeatures():
                new_vector = []
            else:
                new_vector = adjustFeature(new_vector, feature, maxValue - new_vector[feature])
    return new_vector

def adjustFeature(vector : List[int], feature : int, increment : int) -> List[int]:
    # adjust feature (specified by column number) in vector and return the new vector
    # if necessary to maintain ordering contraints, adjust other features as well
    # this method is recursive
    # if it returns an empty vector, the adjustment could not be made because of contraints
    
    # ignore fixed features
    if feature in globals.features.getFixedFeatures():
        return []
    
    
    new_vector = vector.copy()
    new_vector[feature] += increment
    log.debug(f'Adjusting {globals.features.getFeatureName(feature)} by {increment}, old={vector[feature]}, new={new_vector[feature]}')
    
    # get applicable ordered sets
    ordered_sets1 = list(filter(lambda k: feature in k, globals.features.getFeaturesOrderedByHighCards()))
    ordered_sets2 = list(filter(lambda k: feature in k, globals.features.getFeaturesOrderedByLength()))
    assert len(ordered_sets1) == 1, f'No single high-card-based ordered set for {globals.features.getFeatureName(feature)} ({feature})'
    assert len(ordered_sets2) == 1, f'No single length-based ordered set for {globals.features.getFeatureName(feature)} ({feature})'
    ordered_set1 = ordered_sets1[0]
    ordered_set2 = ordered_sets2[0]
    
    # if an increase, bump up next higher feature
    if increment > 0:
        new_vector = bumpUp(new_vector, ordered_set1, ordered_set1.index(feature) + 1, new_vector[feature])
        if new_vector != []:
            new_vector = bumpUp(new_vector, ordered_set2, ordered_set2.index(feature) + 1, new_vector[feature])
    # if a decrease, bummp down next lower feature
    elif increment < 0:
        new_vector = bumpDown(new_vector, ordered_set1, ordered_set1.index(feature) - 1, new_vector[feature])
        if new_vector != []:
            new_vector = bumpDown(new_vector, ordered_set2, ordered_set2.index(feature) - 1, new_vector[feature])
     
    return new_vector

def featuresToIncrement() -> List[int]:
    # if we are ignoring tens, we do not increment features that do have a ten. 
    # Constraints will then ensure that all features with a
    #  ten have the same weight if the ten is missing
    return [x for x in range(globals.features.number_of_features)
            if (not globals.IGNORE_TENS) or 'T' not in globals.features.getFeatureName(x)]
 
def featuresToDecrement() -> List[int]:
    # if we are ignoring tens, we do not decrement features that might have a ten 
    # Constraints will then ensure that all features with a
    #  ten have the same weight if the ten is missing
    return [x for x in range(globals.features.number_of_features)
            if (not globals.IGNORE_TENS) or 'T' in globals.features.getFeatureName(x) or 'x' not in globals.features.getFeatureName(x)]
 

def takeBabyStep(vector : List[int], payoff : float, threshold : np.ndarray,
          targets : pd.core.frame.DataFrame, increment : int) -> Tuple[bool, List[int], float]:
    # Vector is an M-element list of feature weights
    # Payoff is the current best payoff we are trying to beat
    # Thresholds is an N x 1 matrix of thresholds  
    #   It represents the point count for which we predict 3NT is makable.
    # Targets is an N x 2 DataFrame. Which column will be used by the learning algorithm
    #   depends on which payoff function we use
    # Increment is the amount by which we try changes in feature waits
    #
    # The output of this method is a tuple, consisting of a boolean indicating whether
    #   the vector has changed, the new vector, and the accuracy of this new vector
    modified = False
    bestVector = vector.copy()
    bestPayoff = payoff
    
    # vectors_to_try contains an array of vectors we wish to test
    # each column is the original vector with each feature weight in turn either 
    #   incremented or decremented
    # some weights cannot be adjusted because of constraints, so we filter them out
    vectors_to_try = list(filter(lambda v: len(v) > 0, 
        [adjustFeature(bestVector, n, increment) for n in featuresToIncrement()] + 
        [adjustFeature(bestVector, n, -increment) for n in featuresToDecrement()]))
    newPayoffs = calculatePayoff(calculatePointCount(globals.deals.getFeatures(), vectors_to_try),
        threshold, targets)
    
    # see if we have a winner
    if bestPayoff < max(newPayoffs):
        modified = True
        bestPayoff = max(newPayoffs)
        bestVector = list(vectors_to_try[newPayoffs.index(bestPayoff)])
 
        log.info(f'{"Raising" if sum(bestVector) > sum(vector) else "Lowering"} weight ' +
             f'for {globals.vectors.compareVectors(vector, bestVector)}')
        
    return (modified, bestVector, bestPayoff)

def learn(vector : List[int], pointCounts : np.ndarray, threshold : np.ndarray,
          targets : pd.core.frame.DataFrame, starting_increment = STARTING_INCREMENT) -> Tuple[List[int], float]:
    # Vector is an M-element list of feature weights
    # PointCounts is an N x 1 matrix containing the point count for each of N deals
    # Thresholds is an N x 1 matrix of thresholds  
    #   It represents the point count for which we predict 3NT is makable.
    # Targets is an N x 2 DataFrame. Which column will be used by the learning algorithm
    #   depends on which payoff function we use
    # Starting_increment is an optional parameter that permits us to start with an increment
    #   greater than one and reduce the increment as we get closer to a solution
    #
    # The output of this method is a tuple, consisting of a new vector of weights and
    #   the accuracy of that vector
    currentVector = vector.copy()
    currentPayoff = calculatePayoff(pointCounts, threshold, targets)[0]
    log.info(f'iteration 0: vector={currentVector}, payoff={currentPayoff}')

    increment = starting_increment
    for i in range(MAX_ITERATIONS):
        modified, currentVector, currentPayoff = takeBabyStep(currentVector, currentPayoff, 
            threshold, targets, increment)
        if not modified:
            if increment == 1:
                print(f'iteration {i+1}:, No change - finished learning')
                break
            else:
                increment = int((increment + 1) / 2.0)
                print(f'iteration {i+1}:, No change - reducing increment to {increment}')
                continue
        log.info(f'iteration {i+1}:, vector={currentVector}, payoff={currentPayoff}')
    return currentVector, currentPayoff
   
