import logging
import numpy as np
import scipy as sc
import pymc3 as pm
import os
from symbolic_inference import SymbolicInferrer
from os.path import join
from os import mkdir
import time
import statistics
import json
import subprocess
from runtime import Trace, TraceItem
from functools import reduce
from DD import DD
from bin_utils import Bin
from utils import DefectClass as DC
from utils import TraceInfo as TI
from z3 import Select, Concat, Array, BitVecSort, BitVecVal, Solver, BitVec
from typing import List, Tuple, Dict
from enum import Enum
from custom_types import Block, Sample, BitSeq, Proposal, \
    Ebits, EBitsSeq, BlockEbits, Cost, Location, Loc, TestOut, \
    Angel, AngelicPath, TraceFile

logger = logging.getLogger('guided_infer')
pymc3_logger = logging.getLogger('pymc3')


class ExtractionFailure(Exception):
    pass


class LocDDFailure(Exception):
    def __init__(self):
        pass


class EbitsDDFailure(Exception):
    def __init__(self, min_locs, min_init_ebits):
        self.min_locs = min_locs
        self.min_init_ebits = min_init_ebits


class AngelicValsFound(pm.StopSampling):
    def __init__(self, trace_file, unique_trial, total_trial,
                 cost_seq, ratio_seq, accepted_seq):
        self.trace_file = trace_file
        self.unique_trial = unique_trial
        self.total_trial = total_trial
        self.cost_seq = cost_seq
        self.ratio_seq = ratio_seq
        self.accepted_seq = accepted_seq


class SampleSpaceExhausted(pm.StopSampling):
    def __init__(self, sample_space_size, samples, unique_trial, total_trial,
                 cost_seq, ratio_seq, accepted_seq):
        self.sample_space_size = sample_space_size
        self.samples = samples
        self.unique_trial = unique_trial
        self.total_trial = total_trial
        self.cost_seq = cost_seq
        self.ratio_seq = ratio_seq
        self.accepted_seq = accepted_seq


class Stuck(pm.StopSampling):
    def __init__(self, cost, unique_trial, total_trial,
                 cost_seq, ratio_seq, accepted_seq):
        self.cost = cost
        self.unique_trial = unique_trial
        self.total_trial = total_trial
        self.cost_seq = cost_seq
        self.ratio_seq = ratio_seq
        self.accepted_seq = accepted_seq


class EbitsOverFlow(pm.StopSampling):
    def __init__(self, ebits, last_sample, unique_trial, total_trial,
                 cost_seq, ratio_seq, accepted_seq):
        self.ebits = ebits
        self.last_sample = last_sample
        self.unique_trial = unique_trial
        self.total_trial = total_trial
        self.cost_seq = cost_seq
        self.ratio_seq = ratio_seq
        self.accepted_seq = accepted_seq


class TrialsExhausted(pm.StopSampling):
    def __init__(self, unique_trial, total_trial,
                 cost_seq, ratio_seq, accepted_seq):
        self.unique_trial = unique_trial
        self.total_trial = total_trial
        self.cost_seq = cost_seq
        self.ratio_seq = ratio_seq
        self.accepted_seq = accepted_seq


class ChunkOverFlow(Exception):
    def __init__(self, ebits):
        self.ebits = ebits


class CustomProposal:
    """
    s: markov chain
    """
    def __init__(self, kwargs):
        self.project = kwargs['project']
        self.test = kwargs['test']
        self.locations = kwargs['locations']
        self.working_dir = kwargs['working_dir']
        self.searchDir = kwargs['searchDir']
        self.one_bit_flip_prob = kwargs['one_bit_flip_prob']
        self.mul_bits_flip_prob = kwargs['mul_bits_flip_prob']
        self.inferrer = kwargs['inferrer']
        self.config = kwargs['config']
        self.environment = dict(os.environ)

    """
    q0: the current (padded) flattened value
    e.g. if the original sample is [array([0,1]), array([1,0])], then
    q0 becomes array([0, 1, 1, 0])
    return: proposed value
    """
    def __call__(self, q0):
        # logger.debug('[CustomProposal] q0: {}'.format(q0))
        q0 = q0.astype('int64')

        # q0: a flattened 1-d array. This is a concatenation of all chunks.
        # With q0, there is no distinction between locations.
        # e.g., array([   0, 2046,    0,    0])
        #
        # q0_chunks: q0 is split into arrays where each array represents the bitvector for
        # a distinct location.
        # e.g., [array([   0, 2046]), array([0, 0])] for 2 locations
        q0_chunks = list(
            self.inferrer.chunks(np.copy(q0),
                                 self.inferrer.chunks_in_block)) if len(q0) > 0 else []

        # q0_chunks should be idential with the original sample
        # logger.debug('q0_chunks: {}'.format(q0_chunks))
        # logger.debug('chunks_in_block: {}'.format(self.inferrer.chunks_in_block))
        self.inferrer.set_q0(q0_chunks)
        q = list(map(self.propose,
                     zip(q0_chunks, self.inferrer.cur_ebits_list, range(len(q0_chunks)))))
        return q

    """
    propose a new block for each suspicious location
    """
    def propose(self, args: Tuple[np.ndarray, int, int]) -> np.ndarray:
        chunk_bits = self.inferrer.chunk_bits
        q0_i, bits_i, idx = args

        def modifiable_bits_size(chunk_idx):
            if chunk_bits * (len(unpadded) - chunk_idx) <= bits_i:
                return chunk_bits
            else:
                rst = bits_i % chunk_bits
                assert rst != 0
                return rst

        q0_i = np.copy(q0_i)
        act_chunks_in_block = int(np.ceil(bits_i / chunk_bits))
        assert act_chunks_in_block <= len(q0_i)
        unpadded = q0_i[len(q0_i) - act_chunks_in_block:len(q0_i)]
        pad_len = len(q0_i) - len(unpadded)

        if self.config['group_size'] > 1 and \
           self.inferrer.scores[idx] < np.random.rand():
            # As fault location score is lower, we do not change
            # the bits with a higher probability.
            q_i = q0_i
        elif self.choose_maj():
            # flip one bit
            logger.debug('flip 1 bit')
            range = np.arange(len(unpadded))
            if len(range) <= 0:
                q_i = q0_i
            else:
                chunk_idx = np.random.choice(range)
                bits_size = modifiable_bits_size(chunk_idx)
                kth = np.random.choice(np.arange(bits_size))
                unpadded[chunk_idx] = self.flip(unpadded[chunk_idx], kth)
                q_i = Bin.pad_zeros(unpadded, pad_len, 0, int)
        else:
            # flip N bits where N >= 0
            range = np.arange(0, bits_i + 1)
            if len(range) <= 0:
                q_i = q0_i
            else:
                num_of_flips = np.random.choice(range)
                logger.debug('flip {} bits'.format(num_of_flips))
                if num_of_flips == 0:
                    q_i = q0_i
                else:
                    pos_dict = dict()
                    flipped = 0
                    while flipped < num_of_flips:
                        chunk_idx = np.random.choice(np.arange(len(unpadded)))
                        bits_size = modifiable_bits_size(chunk_idx)
                        kth = np.random.choice(np.arange(bits_size))
                        if (chunk_idx, kth) in pos_dict:
                            continue
                        pos_dict.update({(chunk_idx, kth): True})
                        unpadded[chunk_idx] = self.flip(unpadded[chunk_idx], kth)
                        flipped += 1
                    q_i = Bin.pad_zeros(unpadded, pad_len, 0, int)
        return q_i

    def choose_maj(self):
        return np.random.choice([True, False], p=[self.one_bit_flip_prob,
                                                  1 - self.one_bit_flip_prob])

    def flip(self, x, pos):
        return x ^ (1 << pos)


class GuidedInferrer(SymbolicInferrer):

    def __init__(self, config, tester, load, searchDir, dd_dir, extracted, working_dir):
        super().__init__(config, tester, load, searchDir)
        self.dd_dir = dd_dir
        self.extracted = extracted
        self.working_dir = working_dir
        self.one_bit_flip_prob = self.config['one_bit_flip_prob']
        self.mul_bits_flip_prob = 1 - self.one_bit_flip_prob
        self.beta = self.config['mcmc_beta']
        self.max_same_cost_iter = self.config['max_same_cost_iter']
        self.chunk_bits = self.config['chunk_bits']
        self.max_resample = self.config['max_resample']
        self.block_expand_factor = self.config['block_expand_factor']

    def init_sample(self, seed: List[Tuple[str, Loc, str]]) -> Tuple[List[np.ndarray], List[int]]:
        if seed is None:
            try:
                init_sample, cur_ebits_list = self.extract_start()
            except ExtractionFailure as e:
                logger.debug('ExtractionFailure: {}'.format(e))
                logger.warn('failed to extract the inital sample (use the default sample).')
                chunks_ebits_list = list(zip([np.ones(1, dtype=int)] * len(self.c_locs),
                                             [1] * len(self.c_locs)))
                init_sample, cur_ebits_list = self.sample_and_ebits(chunks_ebits_list, init=True)
        else:
            init_sample, cur_ebits_list = self.sample_and_ebit_seq(seed, init=True)
        return init_sample, cur_ebits_list

    def sample(self, init_sample, sample_shape, args_of_proposal_dist):
        if self.config['step_method'] == 'smc':
            logger.warning('smc is not supported')
            exit(1)

            pm.DiscreteUniform('p', 0, 1, shape=sample_shape)
            pm.sample(draws=3,
                      tune=0,
                      compute_convergence_checks=False,
                      cores=1, chains=1,
                      start={'p': init_sample},
                      progressbar=False,
                      logger=pymc3_logger,
                      allow_empty_model=True,
                      step=pm.SMC(likelihood_logp=self.likelihood_logp,
                                  accept_fun=self.accept_fun,
                                  post_accept_fun=self.post_accept_fun,
                                  # S=mc,
                                  proposal_dist=CustomProposal,
                                  random_walk_mc=True,
                                  args_of_proposal_dist=args_of_proposal_dist))
        else:
            # The following dist is given only to trigger sampling.
            # The actual sampling is performed through CustomProposal.
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
                                         proposal_dist=CustomProposal,
                                         random_walk_mc=True,
                                         args_of_proposal_dist=args_of_proposal_dist))

    def get_proposal_file(self, seed, file_name):
        proposal_dir = self.get_proposal_dir()
        proposal_file = join(proposal_dir, file_name + '.json')

        proposal_dict = dict()
        for c_loc in self.c_locs:
            key = reduce((lambda x, y: '{}-{}'.format(x, y)), c_loc)
            vals = [int(TraceItem.get_value(x)) for x in seed]
            proposal_dict[key] = vals

        with open(proposal_file, 'w') as file:
            file.write(json.dumps(proposal_dict))
        return proposal_file

    def repair_cond(self, seed, project, test, locations):
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
                                 'inferrer': self,
                                 'config': self.config}

        logger.debug('seed: {}'.format(seed))
        if len(seed) == 0:
            logger.debug('skip for an empty seed')
            return angelic_paths, ap_trace_file

        init_sample, self.cur_ebits_list = self.init_sample(seed)

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
                    dd_start_time = time.time()
                    seed = Trace.parse_trace_file(e.trace_file)
                    angelic_sample_and_ebit_seq: Tuple[Sample, EBitsSeq] = \
                        self.sample_and_ebit_seq(seed, allow_expand=True)

                    try:
                        if self.config['skip_dd']:
                            raise LocDDFailure

                        seq1 = angelic_sample_and_ebit_seq
                        seq2 = self.init_sample_and_ebit_seq
                        if np.array_equal(seq1[0], seq2[0]) and seq1[1] == seq2[1]:
                            raise LocDDFailure

                        refine = AngelicForestRefine(self, self.project, test, self.environment,
                                                     self.dd_dir, self.locations, self.c_locs,
                                                     self.run_test)
                        angelic_paths, ap_trace_file = refine(angelic_sample_and_ebit_seq,
                                                              self.init_sample_and_ebit_seq,
                                                              e.trace_file)
                    except LocDDFailure:
                        ap_trace_file = e.trace_file
                        angelic_paths = self.get_angelic_paths(ap_trace_file,
                                                               self.c_locs,
                                                               angelic_sample_and_ebit_seq[1])
                        loc_dd_failed = True
                    except EbitsDDFailure as ddf:
                        ap_trace_file = e.trace_file
                        angelic_paths = self.get_angelic_paths(ap_trace_file,
                                                               ddf.min_locs, ddf.min_init_ebits)
                        ebits_dd_failed = True

                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)
                    dd_end_time = time.time()
                    dd_elapsed += dd_end_time - dd_start_time
                    break
                except SampleSpaceExhausted as e:
                    logger.info('Sample space exhausted: size={}, samples={}'.
                                format(e.sample_space_size, e.samples))
                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)
                    sample_space_exhausted = True
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
                except TrialsExhausted as e:
                    logger.info('All {} trials are exhausted'.format(self.search_max_trials))
                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)
                    trials_exhuasted = True
                    break
                except EbitsOverFlow as e:
                    logger.info('Ebits {} is too large'.format(e.ebits))
                    # adjust sampe_shape
                    new_chunks_in_block = \
                        int(np.ceil(e.ebits / self.chunk_bits)) * self.block_expand_factor
                    sample_shape = (sample_shape[0], new_chunks_in_block)

                    # adjust self.chunks_in_block
                    self.init_chunks_in_block(new_chunks_in_block)

                    # adjust init_sample
                    init_sample = [self.pad_chunk(chunk) for chunk in e.last_sample]

                    logger.info('Restart sampling in a larger sample space')
                    logger.info('start vals: {}'.format(init_sample))
                    logger.info('sample shape: {}'.format(sample_shape))

                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)
                    ebits_overflown = True
                    repeat += 1
                    continue

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

    # @profile
    def __call__(self, project, test, locations, dump, score_dict,
                 seed: TraceFile = None, ptr_seed=None) -> Tuple[List[AngelicPath], TraceFile]:
        logger.info('inferring specification for test \'{}\' through guided search'.format(test))

        self.project = project
        self.test = test
        self.dump = dump
        self.score_dict = score_dict
        self.locations = sorted(locations)
        # cond locs
        self.c_locs = [TraceItem.get_location(x) for x in self.locations
                       if DC.is_loop_cond(x[0]) or DC.is_if_cond(x[0])]
        # rhs locs
        self.rhs_locs = [TraceItem.get_location(x) for x in self.locations
                         if DC.is_rhs(x[0])]
        # ptr locs
        self.ptr_locs = [TraceItem.get_location(x) for x in self.locations
                         if DC.is_pointer(x[0])]
        # fault localization scores
        self.scores = [score_dict[loc] for loc in self.locations]
        self.environment = dict(os.environ)
        self.cost_dict = dict()
        self.cur_ebits_list = []
        self.same_cost_count = 0
        self.trial_num = 0
        self.is_first_trial = True
        self.unique_trial = 0
        self.max_sample_space_size = 0
        self.cost_seq = []
        self.accepted_seq = []
        self.ratio_seq = []
        self.search_max_trials = self.config['search_max_trials']

        if len(self.rhs_locs) > 0:
            from klee_inference import KleeInferrer
            rhs_inferrer = KleeInferrer(self)
            rhs_inferrer.init_suspicious_rhses()
            angelic_paths, ap_trace_file = rhs_inferrer.repair_rhs(project, test, dump,
                                                                   locations)
        elif len(self.ptr_locs) > 0:
            from ptr_inference import PtrInferrer
            ptr_inferrer = PtrInferrer(self)
            angelic_paths, ap_trace_file = ptr_inferrer.repair_ptr(ptr_seed, project,
                                                                   test, locations)
        else:
            angelic_paths, ap_trace_file = self.repair_cond(seed, project, test, locations)

        return angelic_paths, ap_trace_file

    def check_sample_space(self):
        # logger.debug('cur_ebits_list: {}'.format(self.cur_ebits_list))
        sample_space_size = reduce(lambda x, y: x * y,
                                   [2**bits for bits in self.cur_ebits_list])
        if sample_space_size > self.max_sample_space_size:
            self.max_sample_space_size = sample_space_size
        keys = self.cost_dict.keys()
        matches = [key for key in keys if list(map(len, key)) == self.cur_ebits_list]

        return self.max_sample_space_size, matches, len(matches) / self.max_sample_space_size

    '''
    run the program with qs
    and returns an acceptance ratio
    '''
    def accept_fun(self, qs, q0s):
        self.trial_num += 1
        logger.info('trial #{}'.format(self.trial_num))

        if len(qs) > 0:
            sample_space_size, samples, usage_rate = self.check_sample_space()
            cur_sample_size = len(samples)
            logger.info('used {}% of sample space'.format(100 * usage_rate))
            if cur_sample_size >= sample_space_size:
                logger.debug('cur_sample_size: {}'.format(cur_sample_size))
                logger.debug('sample_space_size: {}'.format(sample_space_size))
                logger.debug('samples: {}'.format(samples))
                raise SampleSpaceExhausted(sample_space_size, samples,
                                           self.unique_trial, self.trial_num,
                                           self.cost_seq, self.ratio_seq, self.accepted_seq)

        try:
            q0s = self.q0
            self.old_ebits_list = self.cur_ebits_list.copy()
            # logger.debug('[accept_fun] q0s: {} / {} / {}'.format(q0s, self.cur_ebits_list,
            #                                                      np.shape(q0s)))
            # logger.debug('[accept_fun] qs: {} / {} / {}'.format(qs, self.cur_ebits_list,
            #                                                      np.shape(qs)))
            old_cost, _, _ = self.cost(q0s)
            new_cost, new_qs, is_cached_cost = self.cost(qs)
            # logger.debug('new qs: {} / {} / {}'.format(new_qs,
            #                                            self.cur_ebits_list, np.shape(new_qs)))
            self.cost_seq.append(new_cost)
        except ChunkOverFlow as cof:
            raise EbitsOverFlow(cof.ebits, q0s, self.unique_trial, self.trial_num,
                                self.cost_seq, self.ratio_seq, self.accepted_seq)

        # update qs
        for i in range(len(qs)):
            qs[i] = new_qs[i]

        # logger.info('updated qs: {} / {} / {}'.format(qs, self.cur_ebits_list, np.shape(qs)))

        for ebits in self.cur_ebits_list:
            if ebits > np.shape(qs)[1] * self.chunk_bits:
                raise EbitsOverFlow(ebits, q0s, self.unique_trial, self.trial_num,
                                    self.cost_seq, self.ratio_seq, self.accepted_seq)

        if old_cost is None:
            log_ratio = 0
        elif new_cost is None:
            log_ratio = np.log(0.5)
        else:
            if not is_cached_cost and abs(old_cost - new_cost) < self.config['epsilon']:
                if not self.config['always_accept']:
                    if self.same_cost_count >= self.max_same_cost_iter:
                        raise Stuck(new_cost, self.unique_trial, self.trial_num,
                                    self.cost_seq, self.ratio_seq, self.accepted_seq)
                self.same_cost_count += 1

            log_ratio = -self.beta * (new_cost - old_cost)

            log_p_old_to_new = 0
            log_p_new_to_old = 0
            if self.old_ebits_list != self.cur_ebits_list:
                logger.debug('old ebits list: {}'.format(self.old_ebits_list))
                logger.debug('new ebtis list: {}'.format(self.cur_ebits_list))
                for i in range(len(self.cur_ebits_list)):
                    old_chunks = q0s[i]
                    old_bits = self.old_ebits_list[i]
                    new_chunks = qs[i]
                    new_bits = self.cur_ebits_list[i]
                    if old_bits == new_bits:
                        continue
                    else:
                        prob_new_to_old = self.prob_of_transfer(np.copy(new_chunks), new_bits,
                                                                np.copy(old_chunks), old_bits)
                        prob_old_to_new = self.prob_of_transfer(np.copy(old_chunks), old_bits,
                                                                np.copy(new_chunks), new_bits)
                        log_p_new_to_old += prob_new_to_old
                        log_p_old_to_new += prob_old_to_new
                if log_p_old_to_new != log_p_new_to_old:
                    log_ratio = log_ratio + log_p_new_to_old - log_p_old_to_new

            log_ratio = np.minimum(0, log_ratio)
        logger.info('old cost: {}'.format(old_cost))
        logger.info('new cost: {}'.format(new_cost))
        logger.info('all costs: {}'.format(sorted(set(self.cost_dict.values()))))
        logger.info('accept ratio: {}'.format(np.exp(log_ratio)))
        self.ratio_seq.append(np.exp(log_ratio))

        if cur_sample_size == sample_space_size - 1:
            log_ratio = 0

        if self.config['always_accept']:
            # no more choice to make. we do not want to be stuck.
            logger.info('accept. this is the last choice')
            log_ratio = 0

        return log_ratio

    def post_accept_fun(self, accepted):
        self.accepted_seq.append(accepted)
        if not accepted:
            self.cur_ebits_list = self.old_ebits_list

        if self.trial_num >= self.search_max_trials:
            raise TrialsExhausted(self.unique_trial, self.trial_num,
                                  self.cost_seq, self.ratio_seq, self.accepted_seq)

    def likelihood_logp(self, sample):
        return 1

    '''
    Prob that chunk1 is tanferred to chunk2 in the MCMC sample space
    '''
    def prob_of_transfer(self, block1, ebits1, block2, ebits2):
        if ebits1 == ebits2:
            flip_count = Bin.flip_count_between_blocks(block1, block2)
            return self.flip_prob(flip_count, ebits1)
        elif ebits1 < ebits2:
            delta = ebits2 - ebits1
            shifted_block2 = Bin.rshift(block2, delta, self.chunk_bits)
            flip_count = Bin.flip_count_between_blocks(block1, shifted_block2)
            return self.flip_prob(flip_count, ebits1)
        else:
            assert ebits1 > ebits2
            delta = ebits1 - ebits2
            shifted_block1 = Bin.rshift(block1, delta, self.chunk_bits)
            flip_count = Bin.flip_count_between_blocks(shifted_block1, block2)
            p = 0
            for i in range(delta):
                p += sc.special.binom(delta, i) * self.flip_prob(flip_count + i, ebits1)
            return p

    def rshift(self, block, n):
        def get_next_carry(v, n_in_chunk):
            mask_bits = Bin.ones(n_in_chunk)
            return (v & mask_bits) << (self.chunk_bits - n_in_chunk)

        chunk_shift = int(np.floor(n / self.chunk_bits))
        block_copy = np.copy(block[0:len(block) - chunk_shift])
        n_in_chunk = n % self.chunk_bits
        if n_in_chunk != 0:
            cur_carry = 0
            for idx, chunk in enumerate(block):
                next_carry = get_next_carry(chunk, n_in_chunk)
                block_copy[idx] = (chunk >> n_in_chunk) | cur_carry
                cur_carry = next_carry
        return block_copy

    '''
    Prob that 'flips' bits in ebits are flippsed
    '''
    def flip_prob(self, flips, ebits):
        if ebits == 0:
            return self.mul_bits_flip_prob

        if flips == 1:
            # prob that a parituclar bit is flipped
            p = self.one_bit_flip_prob / ebits
            p += self.mul_bits_flip_prob / ((1 + ebits) * ebits)
        else:
            # prob that the number of flipped bits are "flips"
            # possibility: none is flippled, two bits are flipped, ..., "bits" bits are flipped
            p = self.mul_bits_flip_prob / ebits
            p /= (1 + ebits)
            # prob that a particular "flips" bits are flipped
            p /= sc.special.binom(ebits, flips)
        return p

    def flip_count_between_chunks(self, c1, c2):
        min_len = min(len(c1), len(c2))
        if len(c1) != min_len:
            # assert len(c1) > min_len
            start = len(c1) - min_len
            c1 = c1[start:]
        if len(c2) != min_len:
            # assert len(c1) > min_len
            start = len(c2) - min_len
            c2 = c2[start:]

        count = 0
        for i in range(len(c1)):
            count += self.flip_count(c1[i], c2[i])
        return count

    '''
    Return count of bit differences betwee a and b
    '''
    def flip_count(self, a, b):
        def countSetBits(n):
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        rst = countSetBits(a ^ b)
        return rst

    def update_cost(self, sample, cost):
        if cost is not None:
            key = self.sample_to_key(sample, self.cur_ebits_list)
            if key != ('',):
                logger.debug('store the cost of {} [actual]'.format(key))
                self.cost_dict.update({key: cost})

    def update_cost_of_key(self, key, cost):
        if cost is not None:
            if key != ('',):
                logger.debug('store the cost of {} [key]'.format(key))
                self.cost_dict.update({key: cost})

    def cost(self, sample) -> Tuple[Cost, Sample]:
        key: Tuple[str] = self.sample_to_key(sample, self.cur_ebits_list)
        logger.debug('search for the cost of {}'.format(key))
        # check if the sample is already tried before
        if key in self.cost_dict:
            logger.debug('{} cached'.format(key))
            cost = self.cost_dict.get(key)
            act_sample = sample
            is_cached_cost = True
        else:
            act_sample, self.cur_ebits_list, cost = self.new_cost(key, sample)
            is_cached_cost = False
            # TODO: if sample is shorter than actual sample,
            # there is no guarantee on deterministic behavior.
            # better to learn about deterministic behavior.
            # self.update_cost_of_key(key, cost)
            self.update_cost(act_sample, cost)

            # matches = [k for k in self.cost_dict.keys() if self.is_prefix(k, key)]
            # if len(matches) > 0:
            #     logger.debug('{} cached (prefix) by {}'.format(key, matches[0]))
            #     cost = self.cost_dict.get(matches[0])
            #     act_sample = sample
            #     is_cached_cost = True
            # else:
            #     act_sample, self.cur_ebits_list, cost = self.new_cost(key, sample)
            #     is_cached_cost = False
            #     # TODO: if sample is shorter than actual sample,
            #     # there is no guarantee on deterministic behavior.
            #     # better to learn about deterministic behavior.
            #     self.update_cost_of_key(key, cost)
            #     self.update_cost(act_sample, cost)
        return cost, act_sample, is_cached_cost

    '''
    return True if t1 is the prefix of t2
    '''
    def is_prefix(self, t1, t2):
        def _is_prefix(bv, pair):
            return bv and pair[1].startswith(pair[0])

        # logger.debug('t1: {}'.format(t1)
        # logger.debug('t2: {}'.format(t2)
        assert len(t1) == len(t2)
        return reduce(_is_prefix, zip(t1, t2), True)

    def new_cost(self, key, sample):
        logger.debug('[new_cost] key: {}'.format(key))
        # logger.debug('[new_cost] sample: {}'.format(sample))
        sample_bits = list(map(lambda x: [int(x[i:i + 1])
                                          for i in range(0, len(x))], key))
        # logger.debug('[new_cost] sample_bits: {}'.format(sample_bits))
        proposal_file, trace_file, cost_file, act_out_file = self.trial(sample_bits)

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
        self.environment['CC'] = 'angelix-compiler --test' if self.config['use_gcc'] \
            else 'angelix-compiler --klee'

        if self.config['use_gcc']:
            if self.config['use_frontend_for_test']:
                self.environment['CC'] = 'angelix-compiler --frontend'
            else:
                self.environment['CC'] = 'angelix-compiler --test'
        else:
            self.environment['CC'] = 'angelix-compiler --klee'

        try:
            passed = self.run_test(self.project, self.test, env=self.environment)
        except subprocess.TimeoutExpired:
            passed = False
        self.unique_trial += 1
        if passed is True:
            self.cost_seq.append(0)
            raise AngelicValsFound(trace_file, self.unique_trial, self.trial_num,
                                   self.cost_seq, self.ratio_seq, self.accepted_seq)

        try:
            logger.debug('trace_file: {}'.format(trace_file))
            trace = Trace.parse_trace_file(trace_file)
            if len(trace) > self.config['max_bits']:
                logger.warning('actual bits ({}) exceeds the maximum allowed bits ({})'.
                               format(len(trace), self.config['max_bits']))
                act_sample = sample
                ebits_list = self.cur_ebits_list
            elif self.config['fixed_bv_len']:
                act_sample = sample
                ebits_list = self.cur_ebits_list
            else:
                act_sample, ebits_list = self.sample_and_ebit_seq(trace)
        except ExtractionFailure:
            ebits_list = self.cur_ebits_list

        logger.debug('cost file: {}'.format(cost_file))
        cost = self.extract_cost(cost_file)
        logger.debug('extracted cost: {}'.format(cost))
        return act_sample, ebits_list, cost

    def extract_cost(self, cost_file) -> int:
        if self.config['random_cost']:
            range = np.arange(0, self.config['default_max_cost'] + 1)
            return np.random.choice(range)
        if not os.path.exists(cost_file):
            logger.warning('cost file missing: {}'.format(cost_file))
            cost = self.config['default_max_cost']
            return cost
        try:
            cost_txt = subprocess.check_output('cat ' + cost_file, shell=True).decode('ascii')
        except subprocess.CalledProcessError:
            logger.warning('cost file missing: {}'.format(cost_file))
            cost = self.config['default_max_cost']
            return cost

        try:
            cost = float(cost_txt)
        except ValueError:
            cost_txt = cost_txt.replace('\n', '')
            if cost_txt == 'max_cost':
                cost = self.max_cost()
                logger.warning('extract the current max cost: {}'.format(cost))
            else:
                logger.warning('unrecognized cost {} in {}'.format(cost_txt, cost_file))
                cost = None
        return cost

    def max_cost(self):
        return max(self.cost_dict.values(), default=self.config['default_max_cost'])

    def get_proposal_dir(self):
        if not os.path.exists(self.searchDir[self.test]):
            mkdir(self.searchDir[self.test])

        proposal_dir = join(self.searchDir[self.test], 'proposal')
        if not os.path.exists(proposal_dir):
            mkdir(proposal_dir)
        return proposal_dir

    def trial(self, proposal: List[List[int]]):
        logger.debug('proposal: {}'.format(proposal))
        logger.debug('locations: {}'.format(self.c_locs))
        assert len(proposal) == len(self.c_locs)

        proposal_dir = self.get_proposal_dir()
        proposal_file = join(proposal_dir, 'proposal' + str(self.trial_num) + '.json')

        proposal_dict = dict()
        for idx, c_loc in enumerate(self.c_locs):
            key = reduce((lambda x, y: '{}-{}'.format(x, y)), c_loc)
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

    def to_loc_id_str(self, loc_id: tuple):
        reduce((lambda x, y: '{}-{}'.format(x, y)), loc_id)

    def sample_to_key(self, sample, ebits_list):
        def block_to_bin_str(block, ebits):
            act_chunks_in_block = int(np.ceil(ebits / self.chunk_bits))
            unpadded_block = block[len(block) - act_chunks_in_block:len(block)]
            bin = ""
            for idx, chunk in enumerate(unpadded_block):
                if idx == 0:
                    bin += Bin.bin_str(chunk, ebits % self.chunk_bits)
                else:
                    bin += Bin.bin_str(chunk, self.chunk_bits)
            return bin

        return tuple([block_to_bin_str(block, ebits)
                      for block, ebits in zip(sample, ebits_list)])

    def sample_and_ebit_seq(self, trace,
                            init=False,
                            allow_expand=False) -> Tuple[Sample, EBitsSeq]:
        loc_idx = 1
        val_idx = 2

        def _extract(loc: Tuple[int, int, int, int]):
            extracted_bits = [t[val_idx] for t in trace if t[loc_idx] == loc]
            bits = list(map(Bin.normalize_bit, extracted_bits))
            val_chunks = self.bits_to_chunks(bits)
            return val_chunks, len(extracted_bits)

        vals_ebits_list = [_extract(loc) for loc in self.c_locs]
        return self.sample_and_ebits(vals_ebits_list, init=init, allow_expand=allow_expand)

    def extract_start(self) -> Tuple[Sample, EBitsSeq]:
        trace_file = join(self.working_dir, "trace", self.test)
        logger.debug('trace_file: {}'.format(trace_file))
        val_idx = 2

        def _extract(loc):
            pattern = '\"^[IL], ' + reduce((lambda x, y: '{} {}'.format(x, y)), loc) + ',\"'
            try:
                lines = subprocess.check_output('grep ' + pattern + ' ' + trace_file,
                                                shell=True).decode('ascii').splitlines()
            except subprocess.CalledProcessError:
                # lines = []
                raise ExtractionFailure
            extracted_bits = list(map(lambda line: line.split(',')[val_idx].strip(), lines))
            bits = list(map(Bin.normalize_bit, extracted_bits))
            val_chunks = self.bits_to_chunks(bits)
            return np.array(val_chunks), len(lines)

        chunks_ebits_list = [_extract(loc) for loc in self.c_locs]
        logger.debug('chunks_ebits_list: {}'.format(chunks_ebits_list))
        return self.sample_and_ebits(chunks_ebits_list, init=True)

    '''
    Each control sequence consists of an array of bits (i.e., a seq-chunk)
    and the number of enabled bits (i.e., ebits).
    The actual enabled bits are the last N bits of the seq-chunk where N is ebits.
    '''
    def sample_and_ebits(self, block_ebits_list: List[BlockEbits],
                         init=False, allow_expand=False) -> Tuple[Sample, EBitsSeq]:
        block = [pair[0] for pair in block_ebits_list]
        ebits = [pair[1] for pair in block_ebits_list]
        if init:
            if len(block) == 0:
                self.init_chunks_in_block(0)
            else:
                self.init_chunks_in_block(
                    int(np.ceil(max(map(len, block)) * self.block_expand_factor)))

        if allow_expand:
            new_chunks_in_block = max(self.chunks_in_block, max(map(len, block)))
            logger.debug('old chunks_in_block: {}'.format(self.chunks_in_block))
            logger.debug('new chunks_in_block: {}'.format(new_chunks_in_block))
            self.init_chunks_in_block(new_chunks_in_block)

        sample = []
        for idx, chunk in enumerate(block):
            if len(chunk) > self.chunks_in_block:
                logger.info('ChunkOverFlow')
                # logger.info('chunk = {}'.format(chunk))
                logger.info('chunk len = {}'.format(len(chunk)))
                logger.info('chunk size = {}'.format(self.chunks_in_block))
                logger.info('ebits = {}'.format(block_ebits_list[idx][1]))
                assert len(chunk) <= self.config['max_bits']
                raise ChunkOverFlow(block_ebits_list[idx][1])
            else:
                sample.append(self.pad_chunk(chunk))
        return sample, ebits

    def get_angelic_paths(self, trace_file, locs: List[Location], ebits_list: List[Ebits]):
        assert len(locs) == len(ebits_list)

        loc_count_dict: Dict[Location, int] = dict()
        for loc in locs:
            loc_count_dict.update({loc: 0})

        loc_ebits_dict: Dict[Location, int] = dict()
        for idx, loc in enumerate(locs):
            loc_ebits_dict.update({loc: ebits_list[idx]})

        spec_dict = dict()
        with open(trace_file) as f:
            for _, line in enumerate(f):
                try:
                    commas = line.count(', ')
                    if commas == 3:
                        dc, raw_loc, angelic, ctxt = line.split(', ', maxsplit=3)
                    elif commas == 2:
                        dc, raw_loc, angelic = line.split(', ', maxsplit=2)
                        ctxt = None
                    elif commas == 4:
                        dc, raw_loc, angelic, ctxt, rest = line.split(', ', maxsplit=4)
                    else:
                        raise Exception('Ill-formed line: {}'.format(line))
                except ValueError as e:
                    logger.warning('failed to parse line: {}'.format(line))
                    raise e
                loc = Trace.parseLoc(raw_loc)
                if loc not in locs:
                    continue
                if loc_count_dict[loc] >= loc_ebits_dict[loc]:
                    continue
                loc_count_dict.update({loc: loc_count_dict[loc] + 1})
                if spec_dict.get(loc) is None:
                    spec_dict[loc] = [(True if int(angelic) == 1 else False,
                                       None,
                                       Trace.parseCtxt(ctxt))]
                else:
                    spec_dict[loc].append((True if int(angelic) == 1 else False,
                                           None,
                                           Trace.parseCtxt(ctxt)))
        return [spec_dict]

    '''
    The number of elements in a seq-chunk.
    '''
    def init_chunks_in_block(self, size):
        self.chunks_in_block = size

    def pad_chunk(self, chunk):
        assert len(chunk) <= self.chunks_in_block
        if len(chunk) < self.chunks_in_block:
            pad_len = self.chunks_in_block - len(chunk)
            return Bin.pad_zeros(chunk, pad_len, 0, int)
        else:
            return np.array(chunk)

    def bits_to_chunks(self, bits):
        # logger.debug('[bits_to_chunks] bits: {}'.format(bits))
        pad_len = (self.chunk_bits - (len(bits) % self.chunk_bits)) % self.chunk_bits
        padded_bits = Bin.pad_zeros(bits, pad_len, 0, str)
        bits_chunks = list(self.chunks(padded_bits, self.chunk_bits))
        val_chunks = list(map(self.bit_to_val, bits_chunks))
        return val_chunks

    def bit_to_val(self, bits):
        # logger.debug('[bit_to_val] bits: {}'.format(bits))
        bitstr = ''.join(bits)
        val = int(bitstr, 2)
        return val

    '''
    Given a flattened list, return a list of chunks, and each chunk
    has 'size' elements.
    '''
    def chunks(self, lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    def remove_file(self, f):
        if os.path.exists(f):
            os.remove(f)

    def set_q0(self, q0: Sample):
        self.q0 = q0
        if self.is_first_trial:
            q0_copy = []
            for block in q0:
                q0_copy.append(np.copy(block))
            self.init_sample_and_ebit_seq = (q0_copy, self.cur_ebits_list.copy())
            self.is_first_trial = False


class DeltaType(Enum):
    UNDEF = 1
    LOC = 2
    BIT = 3


class AngelicForestRefine(DD):

    def __init__(self, inferrer: GuidedInferrer, project, test_id, environment,
                 dd_dir, locations, c_locs, run_test):
        DD.__init__(self)
        self.inferrer = inferrer
        self.chunk_bits = inferrer.chunk_bits
        self.project = project
        self.test_id = test_id
        self.environment = environment
        self.dd_dir = dd_dir
        self.locations = locations
        self.c_locs = c_locs
        self.run_test = run_test
        self.angelic: List[BlockEbits] = []
        self.init: List[BlockEbits] = []
        self.delta_type = DeltaType.UNDEF
        self.target_loc = -1
        self.last_passing_trace_file = None

    def __call__(self, angelic_sample_and_ebits_list,
                 init_sample_and_ebits_list, trace_file) -> Tuple[List[AngelicPath], TraceFile]:
        self.instance = 0
        self.last_passing_trace_file = trace_file
        angelic: List[BlockEbits] = self.block_ebits_list(angelic_sample_and_ebits_list)
        init: List[BlockEbits] = self.block_ebits_list(init_sample_and_ebits_list)
        assert len(angelic) > 0
        assert len(init) == len(angelic)
        logger.info('angelic: {}'.format(angelic))
        logger.info('init: {}'.format(init))

        self.angelic, self.init = self.adjust_block_size(angelic, init)

        one_minimal_loc = self.dd_locations()
        min_locs = list(map(lambda idx: self.c_locs[idx], one_minimal_loc))
        min_angelic = list(map(lambda idx: self.angelic[idx], one_minimal_loc))
        min_init = list(map(lambda idx: self.init[idx], one_minimal_loc))
        min_init_ebits = list(map(lambda t: t[1], min_init))

        for i in range(len(min_angelic)):
            self.target_loc = one_minimal_loc[i]
            dd_failed, _ = self.dd_ebits(min_angelic[i], min_init[i], one_minimal_loc[i])
            if dd_failed:
                raise EbitsDDFailure(min_locs, min_init_ebits)

        # pass min_init_ebits to prune
        ap = self.inferrer.get_angelic_paths(self.last_passing_trace_file,
                                             min_locs, min_init_ebits)
        return ap, self.last_passing_trace_file

    def adjust_block_size(self, angelic: List[BlockEbits], init: List[BlockEbits]):
        for idx, (angelic_block, angelic_ebits) in enumerate(angelic):
            init_block, init_ebits = init[idx]
            if len(angelic_block) > len(init_block):
                logger.debug('angelic block is bigger than init block')
                init_block = Bin.pad_zeros(init_block,
                                           len(angelic_block) - len(init_block), 0, int)
                init[idx] = (init_block, init_ebits)
        return angelic, init

    def dd_locations(self):
        def compare_block_ebits(pair):
            block_a = pair[0][0]
            ebits_a = pair[0][1]
            block_b = pair[1][0]
            ebits_b = pair[1][1]
            assert len(block_a) == len(block_b)
            logger.debug('[compare_block_ebits] block_a: {}'.format(block_a))
            logger.debug('[compare_block_ebits] block_b: {}'.format(block_b))
            if len(block_a) == 0:
                return (ebits_a == ebits_b)
            else:
                return (block_a == block_b).all() and (ebits_a == ebits_b)

        diffs = list(map(compare_block_ebits, zip(self.angelic, self.init)))
        logger.debug('diffs: {}'.format(diffs))
        all_deltas = []
        for i, cmp in enumerate(diffs):
            if not cmp:
                all_deltas.append(i)
        logger.debug('all_deltas (locs): {}'.format(all_deltas))
        self.delta_type = DeltaType.LOC
        if len(all_deltas) > 0:
            try:
                one_minimal = self.ddmin(all_deltas)
            except Exception as e:
                logger.warning('DD failed (locs): {}'.format(e))
                one_minimal = []
                raise LocDDFailure()
        else:
            one_minimal = []
        logger.debug('one_minimal (locs): {}'.format(one_minimal))
        return one_minimal

    def dd_ebits(self, angelic: BlockEbits, init: BlockEbits, loc_idx: int):

        def diff_block_ebits(block_a: Block, block_b: Block, chunk_bits):
            logger.debug('block_a: {}'.format(block_a))
            logger.debug('block_b: {}'.format(block_b))
            logger.debug('chunk_bits: {}'.format(chunk_bits))

            assert len(block_a) == len(block_b)
            diff_list = []
            diff_idx = 0
            for idx, chunk in enumerate(block_a):
                if block_a[idx] == block_b[idx]:
                    diff_idx += chunk_bits
                else:
                    bin_str_a = Bin.bin_str(block_a[idx], chunk_bits)
                    bin_str_b = Bin.bin_str(block_b[idx], chunk_bits)
                    for idx in range(chunk_bits):
                        if bin_str_a[idx] != bin_str_b[idx]:
                            diff_list.append(diff_idx)
                        diff_idx += 1
            return diff_list

        logger.debug('angelic (bits): {}'.format(angelic))
        logger.debug('init (bits): {}'.format(init))

        angelic_block = angelic[0]
        angelic_ebits = angelic[1]
        init_block = init[0]
        init_ebits = init[1]
        chunk_bits = self.chunk_bits
        self.init_aligned = self.init.copy()

        if angelic_ebits == init_ebits:
            all_deltas = diff_block_ebits(angelic_block, init_block, chunk_bits)
        elif angelic_ebits > init_ebits:
            delta = angelic_ebits - init_ebits
            init_block_shifted = Bin.lshift(init_block, delta, chunk_bits)
            logger.debug('init_block_shifted: {}'.format(init_block_shifted))
            init_block_aligned = Bin.copy_last_bits(angelic_block, delta,
                                                    init_block_shifted, chunk_bits)
            logger.debug('init_block_aligned: {}'.format(init_block_aligned))
            self.init_aligned[loc_idx] = (init_block_aligned, self.angelic[loc_idx][1])
            all_deltas = diff_block_ebits(angelic_block, init_block_aligned, chunk_bits)
        else:
            assert init_ebits > angelic_ebits
            delta = init_ebits - angelic_ebits
            init_block_shifted = Bin.rshift(init_block, delta, chunk_bits)
            all_deltas = diff_block_ebits(angelic_block, init_block_shifted, chunk_bits)
        logger.debug('all_deltas (bits): {}'.format(all_deltas))
        dd_failed = False
        if len(all_deltas) > 0:
            self.delta_type = DeltaType.BIT
            try:
                one_minimal = self.ddmin(all_deltas)
            except Exception:
                logger.warning('DD failed (ebits)')
                dd_failed = True
                one_minimal = []
        else:
            one_minimal = []
        logger.debug('one_minimal (bits): {}'.format(one_minimal))
        return dd_failed, one_minimal

    def _test(self, deltas):
        assert self.delta_type == DeltaType.LOC or self.delta_type == DeltaType.BIT
        if len(deltas) == 0:
            logger.debug('test({}) = FAIL'.format(deltas))
            return self.FAIL

        if self.delta_type == DeltaType.LOC:
            return self.test_for_loc(deltas)
        else:
            return self.test_for_bit(deltas)

    def test_for_bit(self, deltas) -> TestOut:
        logger.debug('deltas (bit): {}'.format(deltas))

        init_copy: List[BlockEbits] = self.init_aligned.copy()
        for delta in deltas:
            init_copy[self.target_loc] = (Bin.copy_bit(self.angelic[self.target_loc][0],
                                                       delta,
                                                       init_copy[self.target_loc][0],
                                                       self.chunk_bits),
                                          init_copy[self.target_loc][1])
        logger.debug('init_copy (bit): {}'.format(init_copy))
        return self.test_common(init_copy, deltas)

    def test_for_loc(self, deltas) -> TestOut:
        logger.debug('deltas (locs): {}'.format(deltas))

        init_copy = self.init.copy()
        for delta in deltas:
            init_copy[delta] = self.angelic[delta]
        logger.debug('init_copy (locs): {}'.format(init_copy))
        return self.test_common(init_copy, deltas)

    def test_common(self, init, deltas) -> TestOut:
        sample = []
        ebits_list = []
        for (block, ebits) in init:
            sample.append(block)
            ebits_list.append(ebits)
        key = self.inferrer.sample_to_key(sample, ebits_list)
        if key in self.inferrer.cost_dict:
            test_rst = self.FAIL
        else:
            matches = [k for k in self.inferrer.cost_dict.keys()
                       if self.inferrer.is_prefix(k, key)]
            if len(matches) > 0:
                test_rst = self.FAIL
            else:
                test_rst = self.invoke_test(key, deltas, self.c_locs)
        logger.debug('test({}) = {}'.format(deltas, test_rst))
        return test_rst

    def invoke_test(self, key, deltas, c_locs):
        proposal = list(map(lambda x: [int(x[i:i + 1])
                                       for i in range(0, len(x))], key))
        proposal_file, trace_file = self.trial(proposal, c_locs)

        self.inferrer.remove_file(trace_file)

        self.environment['ANGELIX_LOAD_JSON'] = proposal_file
        self.environment['ANGELIX_TRACE_AFTER_LOAD'] = trace_file

        passed = self.run_test(self.project, self.test_id, env=self.environment)
        self.instance += 1
        if passed:
            if self.delta_type == DeltaType.BIT:
                self.last_passing_trace_file = trace_file
            return self.PASS
        else:
            return self.FAIL

    def trial(self, proposal, c_locs):
        proposal_dir = join(self.dd_dir[self.test_id], 'proposal')
        if not os.path.exists(proposal_dir):
            mkdir(proposal_dir)
        proposal_file = join(proposal_dir, 'proposal' + '.json')

        proposal_dict = dict()
        for idx, c_loc in enumerate(c_locs):
            key = reduce((lambda x, y: '{}-{}'.format(x, y)), c_loc)
            proposal_dict[key] = proposal[idx]

        with open(proposal_file, 'w') as file:
            file.write(json.dumps(proposal_dict))

        trace_dir = join(self.dd_dir[self.test_id], 'trace')
        if not os.path.exists(trace_dir):
            mkdir(trace_dir)
        trace_file = join(trace_dir, 'trace' + str(self.instance))

        return proposal_file, trace_file

    def block_ebits_list(self, sample_and_ebits_list):
        sample = sample_and_ebits_list[0]
        ebits_list = sample_and_ebits_list[1]
        return list(zip(sample, ebits_list))
