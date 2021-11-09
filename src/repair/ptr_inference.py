import logging
from guided_inference import GuidedInferrer, TrialsExhausted, \
    AngelicValsFound, Stuck
import subprocess
from runtime import Trace, TraceItem
from copy import deepcopy
import os
import numpy as np
import pymc3 as pm
import time
from utils import TraceInfo as TI
from os.path import join
from os import mkdir
import json
import statistics
from functools import reduce
from typing import List, Tuple, Dict
from custom_types import Block, Sample, BitSeq, Proposal, \
    Ebits, EBitsSeq, BlockEbits, Cost, Location, Loc, TestOut, \
    Angel, AngelicPath, TraceFile

logger = logging.getLogger('ptr_infer')
pymc3_logger = logging.getLogger('pymc3')


class PtrInferrer(GuidedInferrer):

    def __init__(self, parent):
        super().__init__(parent.config, parent.run_test, parent.load, parent.searchDir,
                         parent.dd_dir, parent.extracted, parent.working_dir)
        self.test = parent.test
        self.locations = parent.locations
        self.ptr_locs = parent.ptr_locs
        self.search_max_trials = parent.search_max_trials
        self.is_first_trial = parent.is_first_trial
        self.trial_num = parent.trial_num
        self.unique_trial = parent.unique_trial
        self.cost_seq = parent.cost_seq
        self.ratio_seq = parent.ratio_seq
        self.accepted_seq = parent.accepted_seq
        self.cur_ebits_list = parent.cur_ebits_list
        self.project = parent.project
        self.dump = parent.dump
        self.same_cost_count = 0
        self.cost_dict = dict()
        self.environment = deepcopy(parent.environment)

    def sample(self, init_sample, sample_shape, args_of_proposal_dist):
        pm.DiscreteUniform('p', 0, 1, shape=sample_shape)
        pm.sample(self.search_max_trials * 50, tune=0,
                  compute_convergence_checks=False,
                  cores=1, chains=1,
                  start={'p': init_sample},
                  progressbar=False,
                  logger=pymc3_logger,
                  allow_empty_model=True,
                  step=pm.Metropolis(accept_fun=self.accept_fun,
                                     post_accept_fun=self.post_accept_fun,
                                     # S=mc,
                                     proposal_dist=PtrProposal,
                                     random_walk_mc=True,
                                     args_of_proposal_dist=args_of_proposal_dist))

    def repair_ptr(self, ptr_seed, project, test, locations):
        angelic_paths = []
        ap_trace_file = None
        repeat = 0
        explored = 0
        sampled = 0
        cost_seq = []
        ratio_seq = []
        accepted_seq = []
        dd_elapsed = 0
        angelic_found = False
        sample_space_exhausted = False
        stuck = False
        trials_exhuasted = False
        ebits_overflown = False
        loc_dd_failed = False
        ebits_dd_failed = False

        args_of_proposal_dist = {'project': project,
                                 'test': test,
                                 'locations': locations,
                                 'working_dir': self.working_dir,
                                 'searchDir': self.searchDir,
                                 'one_bit_flip_prob': self.one_bit_flip_prob,
                                 'mul_bits_flip_prob': self.mul_bits_flip_prob,
                                 'inferrer': self}

        logger.debug('ptr_seed: {}'.format(ptr_seed))
        init_sample = self.init_sample(ptr_seed)

        sample_shape = np.shape(init_sample)
        logger.info('init sample: {}'.format(init_sample))

        inference_start_time = time.time()
        while repeat < self.max_resample:
            with pm.Model() as model:
                try:
                    self.sample(init_sample, sample_shape, args_of_proposal_dist)
                    # trace.report._run_convergence_checks(trace, model)
                    break
                except AngelicValsFound as e:
                    angelic_found = True
                    logger.info('found an angelic path for test \'{}\''.format(test))
                    logger.debug('trace_file: {}'.format(e.trace_file))
                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)

                    ap_trace_file = e.trace_file
                    angelic_paths = self.get_angelic_paths(ap_trace_file,
                                                           self.ptr_locs)
                    break
                except TrialsExhausted as e:
                    logger.info('All {} trials are exhausted'.format(self.search_max_trials))
                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)
                    trials_exhuasted = True
                    break
                except Stuck as e:
                    logger.info('Stuck in the same cost: {}'.format(e.cost))
                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)
                    stuck = True
                    break

        inference_end_time = time.time()
        inference_elapsed = inference_end_time - inference_start_time
        statistics.data['time']['inference'] += inference_elapsed
        statistics.data['time']['dd'] += dd_elapsed

        iter_stat = dict()
        iter_stat['locations'] = locations
        iter_stat['test'] = test
        iter_stat['time'] = dict()
        iter_stat['time']['mcmc'] = inference_elapsed - dd_elapsed
        iter_stat['time']['dd'] = dd_elapsed
        iter_stat['paths'] = dict()
        iter_stat['paths']['explored'] = explored
        iter_stat['paths']['sampled'] = sampled
        iter_stat['paths']['angelic_found'] = angelic_found
        iter_stat['paths']['angelic'] = len(angelic_paths)
        iter_stat['paths']['sample_space_exhausted'] = sample_space_exhausted
        iter_stat['paths']['trials_exhuasted'] = trials_exhuasted
        iter_stat['paths']['ebits_overflown'] = ebits_overflown
        iter_stat['paths']['loc_dd_failed'] = loc_dd_failed
        iter_stat['paths']['ebits_dd_failed'] = ebits_dd_failed
        iter_stat['paths']['stuck'] = stuck
        iter_stat['paths']['cost'] = cost_seq
        iter_stat['paths']['ratio'] = ratio_seq
        iter_stat['paths']['accepted'] = accepted_seq
        statistics.data['iterations']['guided'].append(iter_stat)
        statistics.save()

        return angelic_paths, ap_trace_file

    def get_angelic_paths(self, trace_file, locs: List[Location]):
        '''
        ctxt: e.g., n = 2 ; x = 1
        return: {'n': 2, 'x': 1}
        '''
        def parseCtxt(ctxt) -> Dict:

            def parseAssignment(a):
                var, val = list(map(lambda x: x.strip(), a.split("=")))
                return var, int(val)

            if ctxt is None:
                return dict()

            assignments = list(map(lambda x: x.strip(), ctxt.split(';')))
            d = dict()
            for a in assignments:
                var, val = parseAssignment(a)
                d.update({var: val})
            return d

        '''
        loc: e.g., 10-10-10-14
        return (10, 10, 10, 14)
        '''
        def parseLoc(loc):
            l1, l2, l3, l4 = loc.split('-', maxsplit=4)
            return (int(l1), int(l2), int(l3), int(l4))

        loc_count_dict: Dict[Location, int] = dict()
        for loc in locs:
            loc_count_dict.update({loc: 0})

        spec_dict = dict()
        with open(trace_file) as f:
            for _, line in enumerate(f):
                try:
                    commas = line.count(', ')
                    if commas == 4:
                        dc, raw_loc, angelic, ctxt, max_idx = line.split(', ', maxsplit=4)
                    elif commas == 3:
                        dc, raw_loc, angelic, max_idx = line.split(', ', maxsplit=3)
                        ctxt = None
                    else:
                        raise Exception('Ill-formed line: {}'.format(line))
                except ValueError as e:
                    logger.warning('failed to parse line: {}'.format(line))
                    raise e
                loc = parseLoc(raw_loc)
                if loc not in locs:
                    continue
                loc_count_dict.update({loc: loc_count_dict[loc] + 1})
                if spec_dict.get(loc) is None:
                    spec_dict[loc] = [(int(angelic),
                                       None,
                                       parseCtxt(ctxt))]
                else:
                    spec_dict[loc].append((int(angelic),
                                           None,
                                           parseCtxt(ctxt)))
        return [spec_dict]

    def init_sample(self, seed: List[Tuple[str, Loc, str, str]]) -> List[Tuple[int, int]]:
        return [(np.random.randint(0, TI.max_val(info)),
                 TI.max_val(info)) for info in seed]

    def accept_fun(self, qs, q0s):
        self.trial_num += 1
        logger.info('trial #{}'.format(self.trial_num))

        q0s = self.q0
        old_cost, _ = self.cost(q0s)
        new_cost, is_cached_cost = self.cost(qs)
        self.cost_seq.append(new_cost)

        if old_cost is None:
            log_ratio = 0
        elif new_cost is None:
            log_ratio = np.log(0.5)
        else:
            if not is_cached_cost and abs(old_cost - new_cost) < self.config['epsilon']:
                if not self.config['always_accept']:
                    logger.debug('same_cost_count: {}'.format(self.same_cost_count))
                    if self.same_cost_count >= self.max_same_cost_iter:
                        raise Stuck(new_cost, self.unique_trial, self.trial_num,
                                    self.cost_seq, self.ratio_seq, self.accepted_seq)
                self.same_cost_count += 1

            log_ratio = -self.beta * (new_cost - old_cost)
        log_ratio = np.minimum(0, log_ratio)
        logger.info('old cost: {}'.format(old_cost))
        logger.info('new cost: {}'.format(new_cost))
        logger.info('all costs: {}'.format(sorted(set(self.cost_dict.values()))))
        logger.info('accept ratio: {}'.format(np.exp(log_ratio)))
        self.ratio_seq.append(np.exp(log_ratio))

        if self.config['always_accept']:
            # no more choice to make. we do not want to be stuck.
            logger.info('accept. this is the last choice')
            log_ratio = 0

        return log_ratio

    def cost(self, sample):
        key: Tuple[str] = self.sample_to_key(sample)
        logger.info('search for the cost of {}'.format(key))
        # check if the sample is already tried before
        if key in self.cost_dict:
            logger.info('{} cached'.format(key))
            cost = self.cost_dict.get(key)
            is_cached_cost = True
        else:
            cost = self.new_cost(key, sample)
            is_cached_cost = False
            self.update_cost_of_key(key, cost)
        return cost, is_cached_cost

    def new_cost(self, key, sample):
        logger.debug('[new_cost] key: {}'.format(key))
        logger.debug('[new_cost] sample: {}'.format(sample))

        sample_indices = [[int(conf[0]) for conf in sample]]
        logger.debug('[new_cost] sample_indices: {}'.format(sample_indices))
        proposal_file, trace_file, cost_file, act_out_file = self.trial(sample_indices)

        self.remove_file(trace_file)
        self.remove_file(cost_file)
        self.remove_file(act_out_file)

        self.environment['ANGELIX_LOAD_JSON'] = proposal_file
        self.environment['ANGELIX_TRACE_AFTER_LOAD'] = trace_file
        self.environment['ANGELIX_COST_FILE'] = cost_file
        self.environment['ANGELIX_ACT_OUT'] = act_out_file
        self.environment['ANGELIX_COMPUTE_COST'] = 'YES'
        self.environment['PENALTY1'] = self.config['penalty1']
        self.environment['PENALTY2'] = self.config['penalty2']
        self.environment['ANGELIX_DEFAULT_NON_ZERO_COST'] = self.config['default_non_zero_cost']
        self.environment['ANGELIX_ERROR_COST'] = self.config['error_cost']
        self.environment['ANGELIX_WARNING_COST'] = self.config['warning_cost']

        try:
            passed = self.run_test(self.project, self.test, env=self.environment)
        except subprocess.TimeoutExpired:
            passed = False
        self.unique_trial += 1
        if passed is True:
            self.cost_seq.append(0)
            raise AngelicValsFound(trace_file, self.unique_trial, self.trial_num,
                                   self.cost_seq, self.ratio_seq, self.accepted_seq)

        logger.debug('cost file: {}'.format(cost_file))
        cost = self.extract_cost(cost_file)
        logger.debug('extracted cost: {}'.format(cost))
        return cost

    def trial(self, proposal: List[List[int]]):
        logger.debug('proposal: {}'.format(proposal))
        logger.debug('locations: {}'.format(self.ptr_locs))
        assert len(proposal) == len(self.ptr_locs)

        proposal_dir = self.get_proposal_dir()
        proposal_file = join(proposal_dir, 'proposal' + str(self.trial_num) + '.json')

        proposal_dict = dict()
        for idx, loc in enumerate(self.ptr_locs):
            key = reduce((lambda x, y: '{}-{}'.format(x, y)), loc)
            proposal_dict[key] = proposal[idx]

        with open(proposal_file, 'w') as file:
            file.write(json.dumps(proposal_dict))

        trace_dir = join(self.searchDir[self.test], 'trace')
        if not os.path.exists(trace_dir):
            mkdir(trace_dir)
        cur_trace_file = join(trace_dir, 'trace' + str(self.trial_num))

        cost_dir = join(self.searchDir[self.test], 'cost')
        if not os.path.exists(cost_dir):
            mkdir(cost_dir)
        cost_file = join(cost_dir, 'cost' + str(self.trial_num))

        act_out_dir = join(self.searchDir[self.test], 'act_out')
        if not os.path.exists(act_out_dir):
            mkdir(act_out_dir)
        act_out_file = join(act_out_dir, 'act_out' + str(self.trial_num))

        return proposal_file, cur_trace_file, cost_file, act_out_file

    def sample_to_key(self, sample: List[np.ndarray]) -> Tuple[str]:
        def _to_str(conf: np.ndarray) -> Tuple[str, str]:
            lst = []
            for num in conf:
                lst.append(str(num))
            return tuple(lst)

        return tuple([_to_str(conf) for conf in sample])

    def post_accept_fun(self, accepted):
        self.accepted_seq.append(accepted)

        if self.trial_num >= self.search_max_trials:
            raise TrialsExhausted(self.unique_trial, self.trial_num,
                                  self.cost_seq, self.ratio_seq, self.accepted_seq)


class PtrProposal:

    def __init__(self, kwargs):
        self.project = kwargs['project']
        self.test = kwargs['test']
        self.locations = kwargs['locations']
        self.working_dir = kwargs['working_dir']
        self.searchDir = kwargs['searchDir']
        self.one_bit_flip_prob = kwargs['one_bit_flip_prob']
        self.mul_bits_flip_prob = kwargs['mul_bits_flip_prob']
        self.inferrer = kwargs['inferrer']
        self.environment = dict(os.environ)

    """
    q0: the current (padded) flattened value
    e.g. if the original sample is [array([0,1]), array([1,0])], then
    q0 becomes array([0, 1, 1, 0])
    return: proposed value
    """
    def __call__(self, q0):
        logger.info('[PtrProposal] q0: {}'.format(q0))
        q0 = q0.astype('int64')

        # q0: a flattened 1-d array. This is a concatenation of all ptr indices and max indices.
        # With q0, there is no distinction between locations.
        # e.g., array([1, 10, 5, 22])
        #
        # q0_split: q0 is split into arrays where each array at an odd index represents
        # the ptr index for a distinct location, and each array at an even index represents
        # the max value for a distinct location.
        # e.g., [array([1]), array([5])] for 2 locations
        q0_split = list(
            self.inferrer.chunks(np.copy(q0), 2)) if len(q0) > 0 else []

        self.inferrer.set_q0(q0_split)
        q = list(map(self.propose,
                     zip(q0_split, range(len(q0_split)))))
        logger.info('[PtrProposal] q: {}'.format(q))
        return q

    def propose(self, args: Tuple[np.ndarray, int]) -> np.ndarray:
        q0_i, loc_idx = args

        q_i = np.copy(q0_i)
        max_i = q0_i[1]
        q_i[0] = np.array([np.random.randint(0, max_i)])
        return q_i
