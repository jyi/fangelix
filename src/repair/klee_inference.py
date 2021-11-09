import logging
import numpy as np
import pymc3 as pm
import statistics
import os
from os.path import join
from os import mkdir
from copy import deepcopy
import time
import json
from functools import reduce
from utils import KleeLocationInfo as KLI
from utils import KleeLocation as KL
from bin_utils import Bin
from runtime import Trace
from utils import LocationInfo as LI
from utils import DefectClass as DC
from utils import NoSmtError, env_log
from symbolic_inference import SymbolicInferrer
from guided_inference import GuidedInferrer, CustomProposal
from guided_inference import ExtractionFailure, TrialsExhausted
from guided_inference import SampleSpaceExhausted, ChunkOverFlow, EbitsOverFlow, Stuck
import subprocess
from typing import List, Tuple, Dict
from custom_types import Block, Sample, BitSeq, Proposal, \
    Ebits, EBitsSeq, BlockEbits, Cost, Location, TestOut, \
    Angel, AngelicPath, TraceFile

logger = logging.getLogger('klee_infer')
pymc3_logger = logging.getLogger('pymc3')


class KleeInferrer(GuidedInferrer):

    def __init__(self, parent):
        super().__init__(parent.config, parent.run_test, parent.load, parent.searchDir,
                         parent.dd_dir, parent.extracted, parent.working_dir)
        self.test = parent.test
        self.locations = parent.locations
        self.search_max_trials = parent.search_max_trials
        self.max_sample_space_size = parent.max_sample_space_size
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

    def trace_file(self, trace_num):
        klee_trace_dir = join(self.searchDir[self.test], 'klee_trace')
        if not os.path.exists(klee_trace_dir):
            os.makedirs(klee_trace_dir, exist_ok=True)
        trace_file = join(klee_trace_dir, 'trace' + str(trace_num))
        return trace_file

    def init_suspicious_rhses(self):
        self.suspicious_rhses = []
        for loc in self.locations:
            self.suspicious_rhses.append(LI.loc_id(loc))

    def sample_to_key(self, sample, ebits_list, klee_locs):
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

        # bit-vector tuple
        bv_tuple = tuple([block_to_bin_str(block, ebits)
                          for block, ebits in zip(sample, ebits_list)])
        key = tuple(zip(klee_locs, bv_tuple))
        return key

    def cost(self, sample, cur_trace, cur_klee_locs) -> Tuple[Cost, Sample]:
        logger.debug('[cost] sample: {}'.format(sample))
        logger.debug('[cost] cur_trace: {}'.format(cur_trace))
        logger.debug('[cost] cur_klee_locs: {}'.format(cur_klee_locs))
        sample_key = self.sample_to_key(sample, self.cur_ebits_list, cur_klee_locs)
        logger.debug('[cost] sample_key: {}'.format(sample_key))
        # check if the sample is already tried before
        logger.debug('[cost] cost_dict: {}'.format(self.cost_dict))
        if sample_key in self.cost_dict:
            logger.info('cached')
            cost = self.cost_dict.get(sample_key)
            fin_sample = sample
            is_cached_cost = True
        else:
            matches = [k for k in self.cost_dict.keys() if self.subsumes(k, sample_key)]
            if len(matches) > 0:
                logger.info('cached (prefix)')
                cost = self.cost_dict.get(matches[0])
                fin_sample = sample
                is_cached_cost = True
            else:
                act_sample, new_cur_ebits_list, cost = self.new_cost(sample_key, sample,
                                                                     cur_trace, cur_klee_locs)
                logger.debug('[cost] sample: {} / {} / {}'.format(sample,
                                                                  self.cur_ebits_list,
                                                                  np.shape(sample)))
                logger.debug('[cost] act_sample: {} / {} / {}'.format(act_sample,
                                                                      new_cur_ebits_list,
                                                                      np.shape(sample)))
                if (len(act_sample) < len(sample)):
                    # we do not shrink the sample space.
                    fin_sample = sample
                else:
                    fin_sample = act_sample
                    self.update_cur_ebits(new_cur_ebits_list)
                logger.debug('[cost] fin_sample: {} / {} / {}'.format(fin_sample,
                                                                      self.cur_ebits_list,
                                                                      np.shape(fin_sample)))

                is_cached_cost = False
                # TODO: if sample is shorter than actual sample,
                # there is no guarantee on deterministic behavior.
                # better to learn about deterministic behavior.
                # self.update_cost_of_key(sample_key, cost)
                self.update_cost(fin_sample, cost)
        return cost, fin_sample, is_cached_cost

    '''
    return True if trace1 subsumes trace2
    example of trace:
    ((('/angelix/tests/assignment-if/.angelix/frontend/test.c', '425'), '0'),
     (('/angelix/tests/assignment-if/.angelix/frontend/test.c', '542'), '1'))
    '''
    def subsumes(self, trace1, trace2):
        # logger.debug('trace1: {}'.format(trace1))
        # logger.debug('trace2: {}'.format(trace2))

        loc2s = set(map(lambda x: x[0], trace2))

        # prepare a dictionary for efficient lookup
        loc_to_bv = dict()
        for elem in trace1:
            loc1 = elem[0]
            if loc1 not in loc2s:
                # loc1 is not found in loc2s.
                # thus, trace1 cannot subsume trace2
                # that is, trace1 is not more general than trace2
                return False
            bv = elem[1]
            loc_to_bv.update({loc1: bv})

        for elem in trace2:
            loc = elem[0]
            bv2 = elem[1]
            bv1 = loc_to_bv.get(loc)
            if bv1 is None:
                continue
            else:
                if not self.is_prefix_of(bv1, bv2):
                    # bv1 is inconsistent with bv2.
                    # thus, trace1 cannot subsume trace2
                    # that is, trace1 is not more general than trace2
                    return False

        return True

    '''
    return True if t1 is the prefix of t2
    '''
    def is_prefix_of(self, t1, t2):
        def _is_prefix(bv, pair):
            return bv and pair[1].startswith(pair[0])

        if len(t1) != len(t2):
            logger.debug('[is_prefix_of] t1: {}'.format(t1))
            logger.debug('[is_prefix_of] t2: {}'.format(t2))
            return False
        return reduce(_is_prefix, zip(t1, t2), True)

    def seen(self, new_cost):
        if new_cost is not None:
            for old_cost in self.cost_dict.values():
                if old_cost is not None and \
                   abs(old_cost - new_cost) < self.config['epsilon']:
                    return True
        return False

    '''
    run the program with qs
    and returns an acceptance ratio
    '''
    def accept_fun(self, qs, q0s):
        self.trial_num += 1
        logger.info('trial #{}'.format(self.trial_num))

        sample_space_size, samples, usage_rate = self.check_sample_space()
        cur_sample_size = len(samples)
        logger.info('used {}% of sample space'.format(100 * usage_rate))

        if cur_sample_size >= sample_space_size:
            raise SampleSpaceExhausted(sample_space_size, samples,
                                       self.unique_trial, self.trial_num,
                                       self.cost_seq, self.ratio_seq, self.accepted_seq)

        try:
            q0s = self.q0
            self.old_ebits_list = self.cur_ebits_list.copy()
            logger.info('[accept_fun] q0s: {} / {} / {}'.format(q0s, self.cur_ebits_list,
                                                                np.shape(q0s)))
            logger.info('[accept_fun] qs: {} / {} / {}'.format(qs, self.cur_ebits_list,
                                                               np.shape(qs)))
            cur_klee_locs = deepcopy(self.klee_locs)
            cur_trace = deepcopy(self.trace)
            # logger.debug('[accept_fun] cur_klee_locs: {}'.format(cur_klee_locs))
            # logger.debug('[accept_fun] cur_trace: {}'.format(cur_trace))
            old_cost, _, _ = self.cost(q0s, cur_trace, cur_klee_locs)
            new_cost, new_qs, is_cached_cost = self.cost(qs, cur_trace, cur_klee_locs)
            logger.info('[accept_fun] new qs: {} / {} / {}'.format(new_qs,
                                                                   self.cur_ebits_list,
                                                                   np.shape(new_qs)))
            if (len(new_qs) > len(qs)):
                logger.warning('Sample space dimesion is too small!!')
                raise KleeLocsOverflow(self.trace, self.klee_locs)
            elif (len(new_qs) < len(qs)):
                pass
            self.cost_seq.append(new_cost)
        except ChunkOverFlow as cof:
            raise EbitsOverFlow(cof.ebits, q0s, self.unique_trial, self.trial_num,
                                self.cost_seq, self.ratio_seq, self.accepted_seq)

        # update qs
        assert len(new_qs) <= len(qs)
        for i in range(max(len(new_qs), len(qs))):
            if i < len(new_qs):
                qs[i] = new_qs[i]

        logger.info('[accept_fun] updated qs: {} / {} / {}'.format(qs,
                                                                   self.cur_ebits_list,
                                                                   np.shape(qs)))

        for ebits in self.cur_ebits_list:
            if ebits > np.shape(qs)[1] * self.chunk_bits:
                raise EbitsOverFlow(ebits, q0s, self.unique_trial, self.trial_num,
                                    self.cost_seq, self.ratio_seq, self.accepted_seq)

        if old_cost is None:
            log_ratio = 0
        elif new_cost is None:
            log_ratio = np.log(0.5)

        if not is_cached_cost and self.seen(new_cost):
            self.same_cost_count += 1
            logger.debug('same_cost_count: {}'.format(self.same_cost_count))
            logger.debug('max_same_cost_iter: {}'.format(self.max_same_cost_iter))
            logger.debug('always_accept: {}'.format(self.config['always_accept']))
            if not self.config['always_accept']:
                if self.same_cost_count >= self.max_same_cost_iter:
                    logger.info('No progress is observed and stop investigation.')
                    raise Stuck(new_cost, self.unique_trial, self.trial_num,
                                self.cost_seq, self.ratio_seq, self.accepted_seq)

        if old_cost is not None and new_cost is not None:
            log_ratio = -self.beta * (new_cost - old_cost)

        log_p_old_to_new = 0
        log_p_new_to_old = 0
        if self.old_ebits_list != self.cur_ebits_list:
            logger.info('old ebits list: {}'.format(self.old_ebits_list))
            logger.info('new ebtis list: {}'.format(self.cur_ebits_list))
            # TODO1: we did not consider a case where a location of qs is dropped in new_qs
            # because the dropped location is not executed.

            # TODO2: better appraoach when |new_qs| > |q0s|?
            for i in range(min(len(new_qs), len(q0s))):
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
        logger.debug('same cost count: {}'.format(self.same_cost_count))
        logger.debug('is_cached_cost: {}'.format(is_cached_cost))
        logger.debug('epsilon: {}'.format(self.config['epsilon']))
        self.ratio_seq.append(np.exp(log_ratio))

        if cur_sample_size == sample_space_size - 1:
            log_ratio = 0

        if self.config['always_accept']:
            # no more choice to make. we do not want to be stuck.
            logger.info('accept. this is the last choice')
            log_ratio = 0

        return log_ratio

    def repair_rhs(self, project, test, dump, locations):
        args_of_proposal_dist = {'project': project,
                                 'test': test,
                                 'locations': locations,
                                 'working_dir': self.working_dir,
                                 'searchDir': self.searchDir,
                                 'one_bit_flip_prob': self.one_bit_flip_prob,
                                 'mul_bits_flip_prob': self.mul_bits_flip_prob,
                                 'inferrer': self,
                                 'config': self.config}
        trace_file = self.trace_file(self.trial_num)

        env = self.get_klee_env(project)
        env['ANGELIX_TRACE_IN_KLEE'] = trace_file

        assert len(self.suspicious_rhses) > 0
        env['ANGELIX_SUSPICIOUS_RHSES'] = ' '.join(self.suspicious_rhses)
        env['ANGELIX_SYMBOLIC_RUNTIME'] = 'on'

        logger.debug('ANGELIX_TRACE_IN_KLEE: {}'.format(env['ANGELIX_TRACE_IN_KLEE']))
        logger.debug('ANGELIX_SUSPICIOUS_RHSES: {}'.format(env['ANGELIX_SUSPICIOUS_RHSES']))

        klee_sp_start_time = time.time()
        if not self.run_klee_sp(project, test, env):
            logger.warning('klee failed')
            # exit(1)
        klee_sp_end_time = time.time()
        klee_sp_elapsed = klee_sp_end_time - klee_sp_start_time

        smt_files = self.get_smt_files(project)
        oracle = self.get_oracle(dump, test)
        angelic_paths = SymbolicInferrer.get_angelic_paths(self, smt_files, oracle, test,
                                                           project)

        statistics.data['time']['klee'] = klee_sp_elapsed
        statistics.data['time']['inference'] = klee_sp_elapsed
        iter_stat = dict()
        iter_stat['locations'] = self.suspicious_rhses
        iter_stat['test'] = test
        iter_stat['time'] = dict()
        iter_stat['time']['klee'] = klee_sp_elapsed
        statistics.data['iterations']['guided'].append(iter_stat)
        statistics.save()

        if len(angelic_paths) == 0 and os.path.exists(trace_file):
            self.handle_trace(trace_file)
            init_sample, self.cur_ebits_list = self.init_sample(self.trace, self.klee_locs)
            sample_shape = np.shape(init_sample)
            logger.info('init sample: {}'.format(init_sample))
            logger.info('sample shape: {}'.format(sample_shape))
            angelic_paths, ap_trace_file = self.search_for_spec(init_sample, sample_shape,
                                                                args_of_proposal_dist, test)

        return angelic_paths, None

    def search_for_spec(self, init_sample, sample_shape, args_of_proposal_dist, test):
        angelic_paths = []
        ap_trace_file = None
        repeat = 0
        explored = 0
        sampled = 0
        dd_elapsed = 0
        cost_seq = []
        ratio_seq = []
        accepted_seq = []
        angelic_found = False
        sample_space_exhausted = False
        stuck = False
        trials_exhuasted = False
        ebits_overflown = False
        loc_dd_failed = False
        ebits_dd_failed = False

        inference_start_time = time.time()
        while repeat < self.max_resample:
            with pm.Model() as model:
                try:
                    self.sample(init_sample, sample_shape, args_of_proposal_dist)
                    break
                except AngelicValsFound as e:
                    angelic_found = True
                    logger.info('found an angelic path for test \'{}\''.format(test))
                    logger.debug('trace_file: {}'.format(e.trace_file))
                    angelic_paths = e.angelic_paths
                    ap_trace_file = e.trace_file
                    seed = Trace.parse_klee_trace_file(e.trace_file)
                    angelic_sample_and_ebit_seq: Tuple[Sample, EBitsSeq] = \
                        self.sample_and_ebit_seq(seed, self.klee_locs, allow_expand=True)

                    explored += e.unique_trial
                    sampled += e.total_trial
                    cost_seq.extend(e.cost_seq)
                    ratio_seq.extend(e.ratio_seq)
                    accepted_seq.extend(e.accepted_seq)
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
                except KleeLocsOverflow as e:
                    logger.info('KleeLocsOverflow occured')
                    # adjust init_sample
                    init_sample, self.cur_ebits_list = self.init_sample(e.trace, e.klee_locs)
                    sample_shape = np.shape(init_sample)
                    logger.info('Restart sampling in a larger sample space')
                    logger.info('new init sample: {}'.format(init_sample))
                    logger.info('new sample shape: {}'.format(sample_shape))
                    repeat += 1
                    continue
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

        iter_stat = dict()
        iter_stat['locations'] = self.suspicious_rhses
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

    def handle_trace(self, trace_file):
        self.trace = Trace.parse_klee_trace_file(trace_file)
        self.klee_locs = self.get_klee_locs(self.trace)

    def get_klee_locs(self, trace):
        #  trace: e.g., [('B', ('/angelix/tests/assignment-if/.angelix/frontend/test.c', '425'), '1'),
        #                ('B', ('/angelix/tests/assignment-if/.angelix/frontend/test.c', '483'), '0')]
        loc_idx = 1
        locs = set()
        for elem in trace:
            locs.add(elem[loc_idx])
        return locs

    def init_sample(self, trace, klee_locs):
        init_sample, ebits_seq = self.sample_and_ebit_seq(trace, klee_locs, init=True)
        return init_sample, ebits_seq

    def sample(self, init_sample, sample_shape, args_of_proposal_dist):
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
                                     proposal_dist=CustomProposalForKlee,
                                     random_walk_mc=True,
                                     args_of_proposal_dist=args_of_proposal_dist))

    def sample_and_ebit_seq(self, trace, klee_locs,
                            init=False,
                            allow_expand=False) -> Tuple[Sample, EBitsSeq]:
        # loc: e.g., ('/angelix/tests/assignment-if/.angelix/frontend/test.c', ' 425')
        def _extract(loc) -> Tuple[List[int], Ebits]:
            extracted_bits = [KLI.value(info) for info in trace
                              if KLI.location(info) == loc and KLI.defect_class(info) == DC.branch_type()]
            bits = list(map(Bin.normalize_bit, extracted_bits))
            val_chunks = self.bits_to_chunks(bits)
            return val_chunks, len(extracted_bits)

        vals_ebits_list = [_extract(loc) for loc in klee_locs]
        return self.sample_and_ebits(vals_ebits_list, init=init, allow_expand=allow_expand)

    def get_proposal_dir(self):
        if not os.path.exists(self.searchDir[self.test]):
            mkdir(self.searchDir[self.test])

        proposal_dir = join(self.searchDir[self.test], 'klee_proposal')
        if not os.path.exists(proposal_dir):
            mkdir(proposal_dir)
        return proposal_dir

    def trial(self, proposal, cur_trace, cur_klee_locs):
        # logger.debug('[trial] proposal: {}'.format(proposal))
        # logger.debug('[trial] cur_trace: {}'.format(cur_trace))
        # logger.debug('[trial] cur_klee_locs: {}'.format(cur_klee_locs))
        assert(len(proposal) <= len(cur_klee_locs))

        proposal_dir = self.get_proposal_dir()
        proposal_file = join(proposal_dir, 'proposal' + str(self.trial_num) + '.json')

        # trace example:
        # [('B', ('/angelix/tests/assignment-if/.angelix/frontend/test.c', ' 425'), '1'),
        #  ('B', ('/angelix/tests/assignment-if/.angelix/frontend/test.c', ' 483'), '0')
        proposal_dict = dict()
        for idx, info in enumerate(cur_klee_locs):
            if idx >= len(proposal):
                logger.warning('Sample space dimesion is too small!!')
                raise KleeLocsOverflow(cur_trace, cur_klee_locs)
            file = KL.file(info)
            if file not in proposal_dict:
                proposal_dict[file] = dict()
            loc = KL.location(info)
            proposal_dict[file][loc] = proposal[idx]

        with open(proposal_file, 'w') as file:
            file.write(json.dumps(proposal_dict))

        cur_trace_file = self.trace_file(self.trial_num)

        cost_dir = join(self.searchDir[self.test], 'cost')
        if not os.path.exists(cost_dir):
            mkdir(cost_dir)
        cost_file = join(cost_dir, 'cost' + str(self.trial_num))

        act_out_dir = join(self.searchDir[self.test], 'act_out')
        if not os.path.exists(act_out_dir):
            mkdir(act_out_dir)
        act_out_file = join(act_out_dir, 'act_out' + str(self.trial_num))

        return proposal_file, cur_trace_file, cost_file, act_out_file

    def update_cost(self, sample, cost):
        logger.debug('[update_cost] cost: {}'.format(cost))
        if cost is not None:
            key = self.sample_to_key(sample, self.cur_ebits_list, self.klee_locs)
            assert key != ('', '', '', '', '')
            logger.debug('[update_cost] store the cost of {}'.format(key))
            self.cost_dict.update({key: cost})

    def update_cost_of_key(self, key, cost):
        if cost is not None:
            logger.debug('[update_cost_of_key] store the cost of {}'.format(key))
            self.cost_dict.update({key: cost})

    def update_cur_ebits(self, new):
        self.cur_ebits_list = new

    def max_cost(self):
        return max(self.cost_dict.values(), default=self.config['default_max_cost'])

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
        logger.info('test({}) = {}'.format(deltas, test_rst))
        return test_rst

    def new_cost(self, sample_key, sample, cur_trace, cur_klee_locs):
        logger.debug('[new_cost] sample_key: {}'.format(sample_key))
        logger.debug('[new_cost] sample: {}'.format(sample))
        # logger.debug('[new_cost] cur_trace: {}'.format(cur_trace))
        # logger.debug('[new_cost] cur_klee_locs: {}'.format(cur_klee_locs))
        # extract bitvectors from sample_key
        # example of sample_key:
        # ((('/angelix/tests/assignment-if/.angelix/frontend/test.c', '425'), '1'),
        #  (('/angelix/tests/assignment-if/.angelix/frontend/test.c', '483'), '0'))
        bvs = map(lambda info: info[1], sample_key)
        sample_bits = list(map(lambda x: [int(x[i:i + 1])
                                          for i in range(0, len(x))], bvs))
        # logger.debug('[new_cost] sample_bits: {}'.format(sample_bits))
        proposal_file, trace_file, cost_file, act_out_file = self.trial(sample_bits,
                                                                        cur_trace, cur_klee_locs)

        self.remove_file(trace_file)
        self.remove_file(cost_file)
        self.remove_file(act_out_file)

        env = self.environment
        env.update(self.get_klee_env(self.project))
        env['ANGELIX_TRACE_IN_KLEE'] = trace_file

        assert len(self.suspicious_rhses) > 0
        env['ANGELIX_SUSPICIOUS_RHSES'] = ' '.join(self.suspicious_rhses)
        env['ANGELIX_PROPOSAL_FOR_KLEE'] = proposal_file
        env['ANGELIX_SYMBOLIC_RUNTIME'] = 'on'

        env['ANGELIX_COST_FILE'] = cost_file
        env['ANGELIX_ACT_OUT'] = act_out_file
        env['ANGELIX_COMPUTE_COST'] = 'YES'
        env['PENALTY1'] = self.config['penalty1']
        env['PENALTY2'] = self.config['penalty2']
        env['ANGELIX_DEFAULT_NON_ZERO_COST'] = self.config['default_non_zero_cost']
        env['ANGELIX_ERROR_COST'] = self.config['error_cost']
        env['ANGELIX_WARNING_COST'] = self.config['warning_cost']

        env_log(logger, 'ANGELIX_TRACE_IN_KLEE', env)
        env_log(logger, 'ANGELIX_SUSPICIOUS_RHSES', env)
        env_log(logger, 'ANGELIX_PROPOSAL_FOR_KLEE', env)
        env_log(logger, 'ANGELIX_SYMBOLIC_RUNTIME', env)
        env_log(logger, 'ANGELIX_COST_FILE', env)
        env_log(logger, 'ANGELIX_ACT_OUT', env)
        env_log(logger, 'ANGELIX_COMPUTE_COST', env)
        env_log(logger, 'ANGELIX_ERROR_COST', env)
        env_log(logger, 'ANGELIX_WARNING_COST', env)

        try:
            passed, angelic_paths = self.run_test_with_klee_sp(self.project, self.test, env)
            logger.debug('run_test_with_klee_sp result: {}'.format(passed))
        except subprocess.TimeoutExpired:
            passed = False
        self.unique_trial += 1
        if passed is True:
            self.cost_seq.append(0)
            raise AngelicValsFound(angelic_paths, trace_file, self.unique_trial, self.trial_num,
                                   self.cost_seq, self.ratio_seq, self.accepted_seq)

        try:
            logger.debug('[new_cost] trace_file: {}'.format(trace_file))
            self.handle_trace(trace_file)
            act_sample, ebits_list = self.sample_and_ebit_seq(self.trace, self.klee_locs)
        except ExtractionFailure:
            ebits_list = self.cur_ebits_list

        logger.debug('[new_cost] cost file: {}'.format(cost_file))
        cost = self.extract_cost(cost_file)
        logger.debug('[new_cost] extracted cost: {}'.format(cost))
        return act_sample, ebits_list, cost

    def run_test_with_klee_sp(self, project, test, environment):
        self.run_klee_sp(project, test, environment)
        try:
            smt_files = self.get_smt_files(project)
            oracle = self.get_oracle(self.dump, test)
            angelic_paths = SymbolicInferrer.get_angelic_paths(self, smt_files, oracle, test,
                                                               project)
            if len(angelic_paths) > 0:
                assert len(angelic_paths) == 1
                return True, angelic_paths
            else:
                return False, None
        except NoSmtError:
            return False, None


class CustomProposalForKlee(CustomProposal):

    """
    propose a new block for each suspicious location
    """
    def propose(self, pair):
        chunk_bits = self.inferrer.chunk_bits
        q0_i, bits_i, idx = pair

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
        if self.choose_maj():
            # flip one bit
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
                logger.debug('[propose] range: {}, num_of_flips: {}'.format(range, num_of_flips))
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
        # logger.debug('[propose] q_i: {}'.format(q_i))
        return q_i


class AngelicValsFound(pm.StopSampling):
    def __init__(self, angelic_paths, trace_file, unique_trial, total_trial,
                 cost_seq, ratio_seq, accepted_seq):
        self.angelic_paths = angelic_paths
        self.trace_file = trace_file
        self.unique_trial = unique_trial
        self.total_trial = total_trial
        self.cost_seq = cost_seq
        self.ratio_seq = ratio_seq
        self.accepted_seq = accepted_seq


class KleeLocsOverflow(pm.StopSampling):
    def __init__(self, trace, klee_locs):
        self.trace = trace
        self.klee_locs = klee_locs
