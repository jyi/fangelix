import os
import stat
from os.path import join, exists, abspath, basename
from os import mkdir
import shutil
import argparse
import time
import json
import logging
import sys
import statistics
from pathlib import Path
import copy
from testing import TestTimeout
from project import Validation, Frontend, Golden, Backend, CompilationError
from utils import format_time, time_limit, TimeoutException, SynthesisFailure
from runtime import Dump, Trace, Load, SearchDir, DeltaDebuggingDir, SynDir
from transformation import RepairableTransformer, SuspiciousTransformer, \
    FixInjector, TransformationError, MutateTransformer
from testing import Tester
from localization import Localizer
from reduction import Reducer
from symbolic_inference import SymbolicInferrer, InferenceError, NoSmtError
from random_inference import RandomInferrer
from guided_inference import GuidedInferrer
from synthesis import Synthesizer
from functools import reduce
from runtime import TraceItem as TI
from multiprocessing import Pool, TimeoutError
from typing import List, Tuple, Dict
from custom_types import Chunk, Block, Sample, BitSeq, Proposal, \
    Ebits, EBitsSeq, BlockEbits, Cost, Location, LocGroup, TestOut, \
    Angel, AngelicPath, TraceFile

logger = logging.getLogger("repair")

SYNTHESIS_LEVELS = ['alternatives',
                    'integer-constants',
                    'boolean-constants',
                    'variables',
                    'basic-arithmetic',
                    'basic-logic',
                    'basic-inequalities',
                    'extended-arithmetic',
                    'extended-logic',
                    'extended-inequalities',
                    'mixed-conditional',
                    'conditional-arithmetic']


DEFECT_CLASSES = ['if-conditions',
                  'assignments',
                  'ptr-assignments',
                  'loop-conditions',
                  'guards']


DEFAULT_DEFECTS = ['if-conditions', 'loop-conditions']


KLEE_SEARCH_STRATEGIES = ['dfs', 'bfs', 'random-state', 'random-path',
                          'nurs:covnew', 'nurs:md2u', 'nurs:depth',
                          'nurs:icnt', 'nurs:cpicnt', 'nurs:qc']

ANGELIC_SEARCH_STRATEGIES = ['random', 'guided', 'symbolic']

STEP_METHODS = ['metropolis', 'smc']

DEFAULT_GROUP_SIZE = 1


DEFAULT_INITIAL_TESTS = 1


sys.setrecursionlimit(10000)  # Otherwise inference.get_vars fails


class Angelix:

    def __init__(self, working_dir, src, buggy, oracle, tests, golden, asserts,
                 lines, build, configure, config):
        self.working_dir = working_dir
        self.config = config
        self.repair_test_suite = tests[:]
        self.validation_test_suite = tests[:]
        self.dump = Dump(working_dir, asserts)
        self.trace = Trace(working_dir)
        self.search_dir = SearchDir(working_dir)
        self.dd_dir = DeltaDebuggingDir(working_dir)
        self.syn_dir = SynDir(working_dir)
        extracted = join(working_dir, 'extracted')
        if exists(extracted):
            shutil.rmtree(extracted, onerror=rm_force)
        os.mkdir(extracted)

        tester = Tester(config, oracle, abspath(working_dir))
        self.run_test = tester
        self.get_suspicious_groups = Localizer(config, lines)
        self.reduce_ts = Reducer(config)
        self.synthesize_fix = Synthesizer(config, extracted, self.syn_dir,
                                          join(working_dir, 'last-angelic-forest.json'))
        if self.config['angelic_search_strategy'] == 'symbolic':
            self.infer_spec = SymbolicInferrer(config, tester, Load(working_dir),
                                               self.search_dir)
        elif self.config['angelic_search_strategy'] == 'random':
            self.infer_spec = RandomInferrer(config, tester, self.search_dir,
                                             extracted, working_dir)
        elif self.config['angelic_search_strategy'] == 'guided':
            self.infer_spec = GuidedInferrer(config, tester, Load(working_dir),
                                             self.search_dir, self.dd_dir,
                                             extracted, working_dir)
        else:
            logger.info('undefined search strategy: {}'.
                        format(self.config['angelic_search_strategy']))
            exit(1)

        self.instrument_for_localization = RepairableTransformer(config, extracted)
        self.instrument_for_inference = SuspiciousTransformer(config, extracted)
        self.apply_patch = FixInjector(config)

        validation_dir = join(working_dir, "validation")
        if not self.config['keep_angelix_dir']:
            logger.info('copy {} to {}'.format(src, validation_dir))
            shutil.copytree(src, validation_dir, symlinks=True)
        self.validation_src = Validation(config, validation_dir, buggy, build, configure)
        self.validation_src.configure()
        compilation_db = self.validation_src.export_compilation_db()
        self.validation_src.import_compilation_db(compilation_db)
        self.validation_src.initialize()

        frontend_dir = join(working_dir, "frontend")
        if not self.config['keep_angelix_dir']:
            shutil.copytree(src, frontend_dir, symlinks=True)
        self.frontend_src = Frontend(config, frontend_dir, buggy, build, configure)
        self.frontend_src.import_compilation_db(compilation_db)
        self.frontend_src.initialize()

        if self.config['angelic_search_strategy'] == 'symbolic':
            backend_dir = join(working_dir, "backend")
            if not self.config['keep_angelix_dir']:
                shutil.copytree(src, backend_dir, symlinks=True)
            self.backend_src = Backend(config, backend_dir, buggy, build, configure)
            self.backend_src.import_compilation_db(compilation_db)
            self.backend_src.initialize()
        else:
            self.backend_src = self.frontend_src

        if golden is not None:
            golden_dir = join(working_dir, "golden")
            if not self.config['keep_angelix_dir']:
                shutil.copytree(golden, golden_dir, symlinks=True)
            self.golden_src = Golden(config, golden_dir, buggy, build, configure)
            self.golden_src.import_compilation_db(compilation_db)
            self.golden_src.initialize()
        else:
            self.golden_src = None

    def filter_impossible(self, negative, env=os.environ):
        if args.golden is None:
            return negative
        self.golden_src.configure()
        self.golden_src.build()
        excluded = []
        for test in negative:
            result = self.run_test(self.golden_src, test, env=env)
            if not result:
                excluded.append(test)

        for test in excluded:
            logger.warning('excluding test {} because it fails in golden version'.
                           format(test))
            negative.remove(test)
            self.repair_test_suite.remove(test)
            self.validation_test_suite.remove(test)
        return negative

    def generate_patch(self):
        env = dict(os.environ)
        env['CC'] = 'angelix-compiler --test' if self.config['use_gcc'] \
            else 'angelix-compiler --klee'
        positive, negative = self.evaluate_ts(self.validation_src,
                                              self.validation_test_suite, env=env)
        negative = self.filter_impossible(negative, env=env)
        positive_traces, negative_traces, \
            suspicious, score_dict = self.fault_localize(positive, negative, env=env)
        if self.config['finish_after_fault_localize']:
            exit(0)

        repaired = len(negative) == 0
        return self.search_patches(suspicious, score_dict, repaired, positive, negative,
                                   positive_traces, negative_traces)

    def search_patches(self, suspicious, score_dict, is_repaired,
                       positive: List[int], negative: List[int],
                       positive_traces, negative_traces):
        if len(suspicious) == 0:
            logger.warning('no suspicious expressions localized')

        patches = []
        while (config['generate_all'] or not is_repaired) and len(suspicious) > 0:
            locations = suspicious.pop(0)
            logger.info('considering suspicious locations {}'.format(locations))
            current_repair_suite = self.reduce_ts(self.repair_test_suite,
                                                  positive_traces, negative_traces, locations)

            if self.config['angelic_search_strategy'] == 'symbolic':
                self.prepare_backend_src(locations)
            else:
                assert self.backend_src == self.frontend_src

            angelic_forest = dict()
            inference_succeed, af_trace_file = self.infer_spec_from_ts(angelic_forest,
                                                                       current_repair_suite,
                                                                       positive, locations,
                                                                       score_dict,
                                                                       positive_traces,
                                                                       negative_traces)
            if not inference_succeed:
                continue

            try:
                is_repaired, pos, neg, \
                    last_level, original, fix = self.try_to_repair(angelic_forest,
                                                                   current_repair_suite)
                logger.debug('is_repaired: {}'.format(is_repaired))
                logger.debug('pos: {}'.format(pos))
                logger.debug('neg: {}'.format(neg))
                logger.debug('last_level: {}'.format(last_level))
                logger.debug('original: {}'.format(original))
                logger.debug('fix: {}'.format(fix))
            except SynthesisFailure:
                continue

            if not is_repaired and last_level == 'variables':
                cur_af = angelic_forest
                while True:
                    red_af = copy.copy(cur_af)
                    for test in cur_af.keys():
                        af_test = cur_af[test]
                        reduced_af_test = []
                        for ap in af_test:
                            reduced_ap = copy.copy(ap)
                            for loc in fix.keys():
                                patch_exp = fix[loc]
                                ap_loc = ap[loc]
                                red_ap_loc = []
                                for inst in ap_loc:
                                    reduced_inst = []
                                    context = inst[2]
                                    reduced_context = dict()
                                    for var in context.keys():
                                        if var not in patch_exp:
                                            reduced_context[var] = context[var]
                                    reduced_inst = (inst[0], inst[1], reduced_context)
                                    red_ap_loc.append(reduced_inst)
                                reduced_ap.update({loc: red_ap_loc})
                            reduced_af_test.append(reduced_ap)
                        red_af[test] = reduced_af_test
                    if red_af == cur_af:
                        break
                    try:
                        logger.debug('angelic_forest: {}'.format(angelic_forest))
                        logger.debug('red_af: {}'.format(red_af))
                        is_repaired, _, _, \
                            last_level, original, fix = self.try_to_repair(red_af,
                                                                           current_repair_suite)
                        if is_repaired:
                            break
                        else:
                            cur_af = red_af
                    except SynthesisFailure:
                        break

            if is_repaired:
                patches.append(self.validation_src.diff_buggy())
                break

            assert not is_repaired
            current_negative = list(set(neg) & set(self.repair_test_suite))
            if len(current_negative) == 0:
                logger.warning("cannot repair using instrumented tests")
                continue

            is_fix_bogus = reduce(lambda x, y: x or y in current_repair_suite,
                                  current_negative, False)
            if is_fix_bogus:
                logger.info('fix is bogus')

                if self.config['inc_fix']:
                    shutil.copy(join(self.validation_src.dir, self.validation_src.buggy),
                                self.frontend_src.dir)
                    positive_traces, negative_traces, suspicious = self.fault_localize(
                        positive, negative)
                    return self.search_patches(suspicious, len(negative) == 0,
                                               positive, negative,
                                               positive_traces, negative_traces)

            patches = self.search_patches_for_counterexamples(patches, angelic_forest,
                                                              locations, score_dict,
                                                              current_repair_suite,
                                                              current_negative, af_trace_file)
        return patches

    def try_to_repair(self, angelic_forest, current_repair_suite):
        max_repair_attempts = self.config['max_repair_attempts']
        for i in range(max_repair_attempts):
            initial_fix, last_level, original = self.synthesize_fix(angelic_forest)
            if initial_fix is None:
                logger.info('cannot synthesize fix')
                raise SynthesisFailure
            logger.info('candidate fix synthesized')

            self.validation_src.restore_buggy()
            try:
                self.apply_patch(self.validation_src, initial_fix)
            except TransformationError:
                logger.info('cannot apply fix')
                raise SynthesisFailure
            self.validation_src.build()

            pos, neg = self.evaluate_ts(self.validation_src, self.validation_test_suite)
            if not set(neg).isdisjoint(set(current_repair_suite)):
                not_repaired = list(set(current_repair_suite) & set(neg))
                logger.warning(
                    'generated invalid fix (tests {} not repaired)'.format(not_repaired))
                continue
            is_repaired = len(neg) == 0
            return is_repaired, pos, neg, last_level, original, initial_fix
        logger.warning('max repair attempts ({}) was exhausted'.format(max_repair_attempts))
        raise SynthesisFailure

    def evaluate_ts(self, src, test_suite, env=os.environ):
        testing_start_time = time.time()

        positive = []
        negative = []
        environment = dict(os.environ)

        def zip_with_src(src, lst):
            return ((src, e) for e in lst)

        if self.config['parallel_testing']:
            with Pool(processes=int(environment['NUM_OF_CPU'])) as pool:
                test_results = pool.starmap(self._run_test, zip_with_src(src, test_suite))
                logger.debug('test_results: {}'.format(test_results))
                for i, result in enumerate(test_results):
                    if result:
                        positive.append(test_suite[i])
                    else:
                        negative.append(test_suite[i])
        else:
            for test in test_suite:
                try:
                    if self.run_test(src, test, env=env, ignore_timeout=True):
                        positive.append(test)
                    else:
                        negative.append(test)
                except TestTimeout:
                    continue

        testing_end_time = time.time()
        testing_elapsed = testing_end_time - testing_start_time
        statistics.data['time']['testing'] += testing_elapsed
        statistics.save()

        return positive, negative

    def _run_test(self, src, test):
        environment = dict(os.environ)
        return self.run_test(src, test, env=environment)

    def prepare_backend_src(self, locations):
        self.backend_src.restore_buggy()
        self.backend_src.configure()
        if config['build_before_instr']:
            self.backend_src.build()
        self.instrument_for_inference(self.backend_src, locations)
        self.backend_src.build()

    def extract_spec(self, test, locations):
        spec_dict = dict()
        trace_file = self.trace[test]
        with open(trace_file) as f:
            for _, line in enumerate(f):
                try:
                    commas = line.count(', ')
                    if commas >= 3:
                        dc, raw_loc, angelic, ctxt = line.split(', ', maxsplit=3)
                    elif commas == 2:
                        dc, raw_loc, angelic = line.split(', ', maxsplit=2)
                        ctxt = None
                    else:
                        raise Exception('Ill-formed line: {} of {}'.
                                        format(line, trace_file))
                except ValueError as e:
                    logger.warning('failed to parse line: {} of {}'.
                                   format(line, trace_file))
                    raise e
                loc = Trace.parseLoc(raw_loc)
                if loc not in locations:
                    continue
                if spec_dict.get(loc) is None:
                    spec_dict[loc] = [(True if int(angelic) == 1 else False,
                                       None,
                                       Trace.parseCtxt(ctxt))]
                else:
                    spec_dict[loc].append((True if int(angelic) == 1 else False,
                                           None,
                                           Trace.parseCtxt(ctxt)))
        return [spec_dict]

    def search_spec(self, locations, score_dict,
                    test, is_positive,
                    seed, ptr_seed=None) -> Tuple[List[AngelicPath], TraceFile]:
        strategy = self.config['angelic_search_strategy']
        if strategy == 'symbolic':
            if self.config['keep_positive_behavior'] and is_positive:
                af = self.extract_spec(test, list(map(lambda x: x[1], locations)))
            else:
                af = self.infer_spec(self.backend_src,
                                     test, locations, self.dump[test], self.frontend_src)
            af_trace_file = None
        elif strategy == 'guided':
            if is_positive:
                af = self.extract_spec(test, list(map(lambda x: x[1], locations)))
                af_trace_file = None
            else:
                af, af_trace_file = self.infer_spec(self.frontend_src, test,
                                                    locations, self.dump[test], score_dict,
                                                    seed=seed, ptr_seed=ptr_seed)
        elif strategy == 'random':
            af = self.infer_spec(self.backend_src,
                                 test, locations)
            af_trace_file = None
        else:
            logger.warning('unknown angelic search strategy: {}'.format(strategy))
            exit(1)
        return af, af_trace_file

    '''
    update angelic_forest
    '''
    def infer_spec_from_ts(self,
                           angelic_forest, current_repair_suite,
                           positive, locations, score_dict,
                           positive_traces, negative_traces) -> Tuple[bool, TraceFile]:
        af_trace_file = None
        locs_only = [TI.get_location(t) for t in locations]
        for test in current_repair_suite:
            try:
                if self.config['spec_from_only_negative']:
                    if test in positive:
                        continue

                traces = negative_traces + positive_traces
                tt = [t[1] for t in traces if t[0] == test][0]
                # interesting test trace
                i_tt = [t for t in tt if TI.get_location(t) in locs_only]
                seed = [t for t in i_tt if TI.is_cond(t)]
                ptr_seed = [t for t in i_tt if TI.is_ptr(t)]
                angelic_forest[test], af_trace_file = self.search_spec(locations, score_dict,
                                                                       test,
                                                                       test in positive,
                                                                       seed=seed,
                                                                       ptr_seed=ptr_seed)
                if len(angelic_forest[test]) == 0:
                    if test in positive:
                        logger.warning(
                            'angelic forest for positive test {} not found'.format(test))
                        current_repair_suite.remove(test)
                        del angelic_forest[test]
                        continue
                    else:
                        return False, af_trace_file
            except InferenceError:
                logger.warning('inference failed (error was raised)')
                return False, af_trace_file
            except NoSmtError:
                logger.warning("no smt file for test {}".format(test))
                if test in positive:
                    current_repair_suite.remove(test)
                    continue
                return False, af_trace_file
        return True, af_trace_file

    def search_patches_for_counterexamples(self, patches, angelic_forest,
                                           locations, score_dict,
                                           current_repair_suite, current_negative,
                                           af_trace_file):
        for counterexample in current_negative:
            logger.info('counterexample test is {}'.format(counterexample))
            current_repair_suite.append(counterexample)
            _, act_out_file = self.prep(counterexample)
            environment = dict(os.environ)
            environment['ANGELIX_ACT_OUT'] = act_out_file

            locs_only = [TI.get_location(t) for t in locations]
            ce_trace = self.trace.parse(counterexample)
            # interesting trace
            i_tt = [t for t in ce_trace if TI.get_location(t) in locs_only]
            seed = [t for t in i_tt if TI.is_cond(t)]

            try:
                angelic_forest[counterexample], _ = \
                    self.search_spec(locations, score_dict,
                                     counterexample,
                                     counterexample not in current_negative,
                                     seed)
            except NoSmtError:
                logger.warning("no smt file for test {}".format(counterexample))
                continue
            if len(angelic_forest[counterexample]) == 0:
                continue
            fix, last_level, original = self.synthesize_fix(angelic_forest)
            if fix is None:
                logger.info('cannot refine fix')
                break
            logger.info('refined fix is synthesized')
            self.validation_src.restore_buggy()
            self.apply_patch(self.validation_src, fix)
            self.validation_src.build()
            pos, neg = self.evaluate_ts(self.validation_src, self.validation_test_suite)
            is_repaired = len(neg) == 0
            if is_repaired:
                patches.append(self.validation_src.diff_buggy())
                return patches

            current_negative = list(set(neg) & set(self.repair_test_suite))
            if not set(current_negative).isdisjoint(set(current_repair_suite)):
                not_repaired = list(set(current_repair_suite) & set(current_negative))
                logger.warning(
                    'generated invalid fix (tests {} not repaired)'.format(not_repaired))
                continue
        return patches

    def fault_localize(self, positive, negative, env=os.environ):
        logger.info('positive: {}'.format(positive))
        logger.info('negative: {}'.format(negative))
        if len(negative) <= 0:
            logger.warning('No negative test!')
            exit(1)
        self.frontend_src.configure()
        if config['build_before_instr']:
            self.frontend_src.build()
        self.instrument_for_localization(self.frontend_src)
        self.frontend_src.build()

        env['ANGELIX_FAULT_LOCALIZE'] = 'YES'

        if self.config['use_gcc']:
            if self.config['use_frontend_for_test']:
                env['CC'] = 'angelix-compiler --frontend'
            else:
                env['CC'] = 'angelix-compiler --test'
        else:
            env['CC'] = 'angelix-compiler --klee'

        testing_start_time = time.time()
        if len(positive) > 0:
            logger.info('running positive tests for debugging')
        for test in positive:
            self.trace += test
            self.dump.add_test(test)
            _, instrumented = self.run_test(self.frontend_src, test,
                                            dump=self.dump[test],
                                            env=env,
                                            trace=self.trace[test],
                                            check_instrumented=True)
            if not instrumented and not self.config['ignore_instrument']:
                logger.debug(
                    'remove {} from repair test suite (not instrumented)'.format(test))
                self.repair_test_suite.remove(test)

        golden_is_built = False
        excluded = []

        if len(negative) > 0:
            logger.info('running negative tests for debugging')
        for test in negative:
            self.trace += test
            _, act_out_file = self.prep(test)
            env['ANGELIX_ACT_OUT'] = act_out_file

            _, instrumented = self.run_test(self.frontend_src, test, env=env,
                                            trace=self.trace[test],
                                            check_instrumented=True)
            if not instrumented and not self.config['ignore_instrument']:
                logger.debug(
                    'remove a negative test {} from repair test suite (not instrumented)'.
                    format(test))
                self.repair_test_suite.remove(test)
            if test not in self.dump:
                # the expected output of test is not defined by the assert file.
                # note that the contents of assert file is encoded into dump.
                if self.golden_src is None:
                    if self.config['angelic_search_strategy'] == 'guided':
                        for dc in self.config['defect']:
                            if dc == 'assignments' or dc == 'ptr-assignments':
                                logger.error("golden version or assert file needed for test {}".
                                             format(test))
                                return []
                        continue
                    else:
                        logger.error("golden version or assert file needed for test {}".
                                     format(test))
                        return []
                if not golden_is_built:
                    self.golden_src.configure()
                    self.golden_src.build()
                    golden_is_built = True
                self.dump += test
                result = self.run_test(self.golden_src, test, env=env, dump=self.dump[test])
                if not result:
                    excluded.append(test)

        for test in excluded:
            if not self.config['mute_test_message']:
                logger.warning('excluding test {} because it fails in golden version'.
                               format(test))
            negative.remove(test)
            if test in self.repair_test_suite:
                logger.debug(
                    'remove an excluded test {} from repair test suite (not instrumented)'.
                    format(test))
                self.repair_test_suite.remove(test)
            self.validation_test_suite.remove(test)

        testing_end_time = time.time()
        testing_elapsed = testing_end_time - testing_start_time
        statistics.data['time']['testing'] += testing_elapsed
        statistics.save()

        logger.info("repair test suite: {}".format(self.repair_test_suite))
        logger.info("validation test suite: {}".format(self.validation_test_suite))

        positive_traces = [(test, self.trace.parse(test)) for test in positive]
        negative_traces = [(test, self.trace.parse(test)) for test in negative]
        suspicious, \
            score_dict = self.get_suspicious_groups(self.validation_test_suite,
                                                    positive_traces, negative_traces)
        return positive_traces, negative_traces, suspicious, score_dict

    def dump_outputs(self, env=os.environ):
        self.frontend_src.configure()
        if config['build_before_instr']:
            self.frontend_src.build()
        self.instrument_for_localization(self.frontend_src)
        self.frontend_src.build()
        logger.info('running tests for dumping')
        for test in self.validation_test_suite:
            self.dump += test
            result = self.run_test(self.frontend_src, test, env=env, dump=self.dump[test])
            if result:
                logger.info('test passed')
            else:
                logger.info('test failed')
        return self.dump.export()

    def synthesize_from(self, af_file):
        with open(af_file) as file:
            data = json.load(file)
        repair_suite = data.keys()

        expressions = set()
        for _, paths in data.items():
            for path in paths:
                for value in path:
                    expr = tuple(map(int, value['expression'].split('-')))
                    expressions.add(expr)

        # we need this to extract buggy expressions:
        self.backend_src.restore_buggy()
        self.backend_src.configure()
        if config['build_before_instr']:
            self.backend_src.build()
        self.instrument_for_inference(self.backend_src, list(expressions))

        fix, last_level, original = self.synthesize_fix(af_file)
        if fix is None:
            logger.info('cannot synthesize fix')
            return []
        logger.info('fix is synthesized')

        self.validation_src.restore_buggy()
        self.apply_patch(self.validation_src, fix)
        self.validation_src.build()
        positive, negative = self.evaluate_ts(self.validation_src, self.validation_test_suite)
        if not set(negative).isdisjoint(set(repair_suite)):
            not_repaired = list(set(repair_suite) & set(negative))
            logger.warning("generated invalid fix (tests {} not repaired)".format(not_repaired))
            return []

        if len(negative) > 0:
            logger.info("tests {} fail".format(negative))
            return []
        else:
            return [self.validation_src.diff_buggy()]

    def prep(self, test):
        if not os.path.exists(self.search_dir[test]):
            mkdir(self.search_dir[test])

        if not os.path.exists(self.dd_dir[test]):
            mkdir(self.dd_dir[test])

        cost_dir = join(self.search_dir[test], 'cost')
        if not os.path.exists(cost_dir):
            mkdir(cost_dir)
        cost_file = join(cost_dir, 'cost0')

        act_out_dir = join(self.search_dir[test], 'act_out')
        if not os.path.exists(act_out_dir):
            mkdir(act_out_dir)
        act_out_file = join(act_out_dir, 'act_out0')

        return cost_file, act_out_file


if __name__ == "__main__":

    def repair():
        if args.synthesis_only is not None:
            return tool.synthesize_from(args.synthesis_only)
        else:
            return tool.generate_patch()

    def rm_force(action, name, exc):
        os.chmod(name, stat.S_IREAD)
        shutil.rmtree(name)

    def is_subdir(path, directory):
        p = Path(os.path.abspath(path))
        d = Path(directory)
        return p in [d] + [p for p in d.parents]

    def mutate(config: Dict, src_file, working_dir, golden_dir, build, configure):
        mutate_dir = join(working_dir, "mutate")
        shutil.copytree(golden_dir, mutate_dir, symlinks=True)

        mutate_src = Validation(config, mutate_dir, src_file, build, configure)
        mutate_src.configure()
        compilation_db = mutate_src.export_compilation_db()
        mutate_src.import_compilation_db(compilation_db)
        mutate_src.initialize()

        extracted = join(working_dir, 'extracted')
        mutate_trasnformer = MutateTransformer(config, extracted)
        mutate_trasnformer(mutate_src)

    parser = argparse.ArgumentParser('angelix')
    parser.add_argument('src', metavar='SOURCE', help='source directory')
    parser.add_argument('buggy', metavar='BUGGY', help='relative path to buggy file')
    parser.add_argument('oracle', metavar='ORACLE', help='oracle script')
    parser.add_argument('tests', metavar='TEST', nargs='+', help='test case')
    parser.add_argument('--golden', metavar='DIR', help='golden source directory')
    parser.add_argument('--assert', metavar='FILE', help='assert expected outputs')
    parser.add_argument('--defect', metavar='CLASS', nargs='+',
                        default=DEFAULT_DEFECTS,
                        choices=DEFECT_CLASSES,
                        help='defect classes (default: %(default)s). choices: ' + ', '.join(DEFECT_CLASSES))
    parser.add_argument('--lines', metavar='LINE', type=int, nargs='+', help='suspicious lines (default: all)')
    parser.add_argument('--configure', metavar='CMD', default=None,
                        help='configure command in the form of shell command (default: %(default)s)')
    parser.add_argument('--build', metavar='CMD', default='make -e',
                        help='build command in the form of simple shell command (default: %(default)s)')
    parser.add_argument('--build-before-instr', action='store_true',
                        help='build source before (and after) instrumentation (default: %(default)s)')
    parser.add_argument('--instr-printf', metavar='FILE', default=None, help='instrument printf arguments as outputs')
    parser.add_argument('--timeout', metavar='SEC', type=int, default=None,
                        help='[deprecated] total repair timeout (default: %(default)s)')
    parser.add_argument('--initial-tests', metavar='NUM', type=int, default=DEFAULT_INITIAL_TESTS,
                        help='initial repair test suite size (default: %(default)s)')
    parser.add_argument('--all-tests', action='store_true',
                        help='use all tests for repair (default: %(default)s)')
    parser.add_argument('--test-timeout', metavar='SEC', type=int, default=None,
                        help='test case timeout (default: %(default)s)')
    parser.add_argument('--group-size', metavar='NUM', type=int, default=DEFAULT_GROUP_SIZE,
                        help='number of statements considered at once (default: %(default)s)')
    parser.add_argument('--single-group', action='store_true',
                        help='use a single group (default: %(default)s)')
    parser.add_argument('--group-by-score', action='store_true',
                        help='group statements by suspiciousness score (default: grouping by location)')
    parser.add_argument('--localize-from-bottom', action='store_true',
                        help='iterate suspicious expression from the bottom of file (default: localizing from top)')
    parser.add_argument('--suspicious', metavar='NUM', type=int, default=20,
                        help='total number of suspicious statements (default: %(default)s)')
    parser.add_argument('--localization', default='jaccard', choices=['jaccard', 'ochiai', 'tarantula'],
                        help='formula for localization algorithm (default: %(default)s)')
    parser.add_argument('--ignore-trivial', action='store_true',
                        help='ignore trivial expressions: variables and constants (default: %(default)s)')
    parser.add_argument('--spec-from-only-negative', action='store_true',
                        help='default: %(default)s')
    parser.add_argument('--path-solving-timeout', metavar='MS', type=int, default=60000, # 60 seconds
                        help='timeout for extracting single angelic path (default: %(default)s)')
    parser.add_argument('--max-angelic-paths', metavar='NUM', type=int, default=None,
                        help='max number of angelic paths for a test case (default: %(default)s)')
    parser.add_argument('--klee-search', metavar='HEURISTIC', default=None,
                        choices=KLEE_SEARCH_STRATEGIES,
                        help='KLEE search heuristic (default: KLEE\'s default). choices: ' + ', '.join(KLEE_SEARCH_STRATEGIES))
    parser.add_argument('--klee-max-forks', metavar='NUM', type=int, default=None,
                        help='KLEE max number of forks (default: %(default)s)')
    parser.add_argument('--klee-max-depth', metavar='NUM', type=int, default=None,
                        help='KLEE max symbolic branches (default: %(default)s)')
    parser.add_argument('--klee-timeout', metavar='SEC', type=int, default=None,
                        help='KLEE timeout (default: %(default)s)')
    parser.add_argument('--klee-out-dir-timeout', metavar='SEC', type=int, default=60,
                        help='time to wait for klee to generate klee_out dir (default: %(default)s)')
    parser.add_argument('--klee-solver-timeout', metavar='SEC', type=int, default=None,
                        help='KLEE solver timeout (default: %(default)s)')
    parser.add_argument('--klee-debug', action='store_true',
                        help='print instructions executed by KLEE (default: %(default)s)')
    parser.add_argument('--klee-ignore-errors', action='store_true',
                        help='Don\'t terminate on memory errors (default: %(default)s)')
    parser.add_argument('--ignore-trans-errors', action='store_true',
                        help='Don\'t terminate on transformation errors (default: %(default)s)')
    parser.add_argument('--ignore-infer-errors', action='store_true',
                        help='Consider path with errors for inference (default: %(default)s)')
    parser.add_argument('--ignore-z3-exception', action='store_true',
                        help='Ignore z3 exception (default: %(default)s)')
    parser.add_argument('--skip-validating-angelic-path', action='store_false',
                        help='Skip validating an angelic path (default: %(default)s)')
    parser.add_argument('--use-nsynth', action='store_true',
                        help='use new synthesizer (default: %(default)s)')
    parser.add_argument('--use-osynth', action='store_true',
                        help='use old synthesizer (default: %(default)s)')
    parser.add_argument('--use-gcc', action='store_true',
                        help='use gcc instead of llvm-gcc (default: %(default)s)')
    parser.add_argument('--use-frontend-for-test', action='store_true',
                        help='default: %(default)s')
    parser.add_argument('--keep-positive-behavior', action='store_true',
                        help='keep the original behavior for positive tests (default: %(default)s)')
    parser.add_argument('--synthesis-timeout', metavar='MS', type=int, default=30000, # 30 sec
                        help='synthesis timeout (default: %(default)s)')
    parser.add_argument('--synthesis-levels', metavar='LEVEL', nargs='+',
                        choices=SYNTHESIS_LEVELS,
                        default=['alternatives', 'integer-constants', 'boolean-constants'],
                        help='component levels (default: %(default)s). choices: ' + ', '.join(SYNTHESIS_LEVELS))
    parser.add_argument('--synthesis-global-vars', action='store_true',
                        help='use global program variables for synthesis (default: %(default)s)')
    parser.add_argument('--synthesis-func-params', action='store_true',
                        help='use function parameters as variables for synthesis (default: %(default)s)')
    parser.add_argument('--synthesis-used-vars', action='store_true',  # for backward compatibility
                        help='[deprecated] use variables that are used in scope for synthesis (default: True)')
    parser.add_argument('--synthesis-ptr-vars', action='store_true',
                        help='use pointer variables for synthesis (default: %(default)s)')
    parser.add_argument('--forced-to-use-bool', action='store_true',
                        help='(default: %(default)s)')
    parser.add_argument('--empty-env-exps', action='store_true',
                        help='empty environment expressions (default: %(default)s)')
    parser.add_argument('--exclude-member-exp', action='store_true',
                        help='exclude member expressions from synthesis (default: %(default)s)')
    parser.add_argument('--generate-all', action='store_true',
                        help='generate all patches (default: %(default)s)')
    parser.add_argument('--init-uninit-vars', action='store_true',
                        help='initialize the uninitialized variables of the program with default values (default: %(default)s)')
    parser.add_argument('--synthesis-bool-only', action='store_true',
                        help='synthesize only boolean expressions (default: %(default)s)')
    parser.add_argument('--max-z3-trials', metavar='NUM', type=int, default=2,
                        help='maxium Z3 trials when using SemFix synthesizer (default: %(default)s)')
    parser.add_argument('--dump-only', action='store_true',
                        help='dump actual outputs for given tests (default: %(default)s)')
    parser.add_argument('--synthesis-only', metavar="FILE", default=None,
                        help='synthesize and validate patch from angelic forest (default: %(default)s)')
    parser.add_argument('--invalid-localization', action='store_true',
                        help='[deprecated] use tests that fail in golden version for localization (default: %(default)s)')
    parser.add_argument('--verbose', action='store_true',
                        help='print compilation and KLEE messages (default: %(default)s)')
    parser.add_argument('--mute-config-message', action='store_true',
                        help='mute configure message (default: %(default)s)')
    parser.add_argument('--mute-build-message', action='store_true',
                        help='mute build message (default: %(default)s)')
    parser.add_argument('--quiet', action='store_true',
                        help='print only errors (default: %(default)s)')
    parser.add_argument('--gobble-klee-message', action='store_true',
                        help='Gobble klee message (default: %(default)s)')
    parser.add_argument('--mute-test-message', action='store_true',
                        help='mute test message (default: %(default)s)')
    parser.add_argument('--show-test-message', action='store_true',
                        help='show test message (default: %(default)s)')
    parser.add_argument('--show-oracle-contents', action='store_true',
                        help='show the contents of the oracle (default: %(default)s)')
    parser.add_argument('--show-syn-message', action='store_true',
                        help='show synthesis message (default: %(default)s)')
    parser.add_argument('--mute-warning', action='store_true',
                        help='mute warning message (default: %(default)s)')
    parser.add_argument('--ignore-lines', action='store_true',
                        help='[deprecated] ignore --lines options (default: %(default)s)')
    parser.add_argument('--ignore-instrument', action='store_true',
                        help='default: %(default)s')
    parser.add_argument('--ignore-unmatched-execution', action='store_true',
                        help='default: %(default)s')
    parser.add_argument('--all-suspicious', action='store_true',
                        help='consider all suspicious locations (default: %(default)s)')
    parser.add_argument('--show-suspicious-locations', action='store_true',
                        help='show all suspicious locations and their scores \
                        (default: %(default)s)')
    parser.add_argument('--tests-summary', action='store_true',
                        help='run validation and golden tests and summarize the tests (default: %(default)s)')
    parser.add_argument('--compilation-db-file',
                        help='Use the provided compilation db file')
    parser.add_argument('--keep-angelix-dir', action='store_true',
                        help='keep .angelix dir (default: %(default)s)')
    parser.add_argument('--skip-configure', action='store_true',
                        help='skip configure (default: %(default)s)')
    parser.add_argument('--skip-build', action='store_true',
                        help='skip build (default: %(default)s)')
    parser.add_argument('--angelic-search-strategy', metavar='STRATEGY', default='guided',
                        choices=ANGELIC_SEARCH_STRATEGIES,
                        help='angelic search strategy. choices: '
                        + ', '.join(ANGELIC_SEARCH_STRATEGIES))
    parser.add_argument('--step-method', metavar='STEP', default='metropolis',
                        choices=STEP_METHODS,
                        help='step method. choices: '
                        + ', '.join(STEP_METHODS))
    parser.add_argument('--search-max-trials', metavar='NUM', type=int, default=100,
                        help='max number of search trials (default: %(default)s)')
    parser.add_argument('--max-same-cost-iter', metavar='NUM', type=int, default=20,
                        help='possible max iteration of the same cost (default: %(default)s)')
    parser.add_argument('--one-bit-flip-prob', metavar='NUM', type=float, default=0.5,
                        help='probability that one bit flips (default: %(default)s)')
    parser.add_argument('--mcmc-beta', metavar='NUM', type=float, default=0.8,
                        help='MCMC beta (default: %(default)s)')
    parser.add_argument('--chunk-bits', metavar='NUM', type=int, default=32,
                        help='the number of bits for a chunk (default: %(default)s)')
    parser.add_argument('--max-bits', metavar='NUM', type=int, default=100,
                        help='max number of bits (default: %(default)s)')
    parser.add_argument('--max-resample', metavar='NUM', type=int, default=3,
                        help='max resample (default: %(default)s)')
    parser.add_argument('--block-expand-factor', metavar='NUM', type=float, default=2,
                        help='block expand factor (default: %(default)s)')
    parser.add_argument('--inc-fix', action='store_true', default=False,
                        help='allow incremental fix (default: %(default)s)')
    parser.add_argument('--fixed-bv-len', action='store_true', default=False,
                        help='use a fixed the bitvector length (default: %(default)s)')
    parser.add_argument('--max-syn-attempts', metavar='NUM', type=float, default=2,
                        help='maximum synthesis attempts (default: %(default)s)')
    parser.add_argument('--max-repair-attempts', metavar='NUM', type=float, default=2,
                        help='maximum repair attempts for each location group \
                        (default: %(default)s)')
    parser.add_argument('--default-max-cost', metavar='NUM', type=float, default=2,
                        help='default max cost (default: %(default)s)')
    parser.add_argument('--error-cost', metavar='NUM', type=str, default="0",
                        help='error cost (default: %(default)s)')
    parser.add_argument('--warning-cost', metavar='NUM', type=str, default="0",
                        help='warning cost (default: %(default)s)')
    parser.add_argument('--penalty1', metavar='NUM', type=str, default="1",
                        help='penalty1 (default: %(default)s)')
    parser.add_argument('--penalty2', metavar='NUM', type=str, default="1",
                        help='penalty2 (default: %(default)s)')
    parser.add_argument('--default-non-zero-cost', metavar='NUM', type=str, default="1",
                        help='Used when test fails to generate non-zero cost. \
                        (default: %(default)s)')
    parser.add_argument('--timeout-cost', metavar='NUM', type=str, default="1",
                        help='cost used when timeout occurs. \
                        (default: %(default)s)')
    parser.add_argument('--log', metavar='LOG', default=None,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='set the logging level')
    parser.add_argument('--parallel-testing', action='store_true',
                        help='perform testing in parallel (default: %(default)s)')
    parser.add_argument('--mutate', metavar='NUM', type=str, default="0",
                        help='mutate the golden version (default: %(default)s)')
    parser.add_argument('--epsilon', metavar='NUM', type=float, default="1",
                        help='cost difference less than epsilon is considered the same.\
                        (default: %(default)s)')
    parser.add_argument('--default-susp-score', metavar='NUM', type=float, default="0.5",
                        help='default suspiciousness score.\
                        (default: %(default)s)')
    parser.add_argument('--additional-susp-locs', metavar='DC:NUM-NUM-NUM-NUM',
                        type=str, nargs='+', default=None,
                        help='additional suspicious locations.(default: %(default)s)')
    parser.add_argument('--always-accept', action='store_true',
                        help='always accept a proposal in MCMC (default: %(default)s)')
    parser.add_argument('--random-cost', action='store_true',
                        help='use random cost in MCMC. (default: %(default)s)')
    parser.add_argument('--max-random-cost', metavar='NUM', type=str, default="0",
                        help='maxium random cost. (default: %(default)s)')
    parser.add_argument('--skip-dd', action='store_true',
                        help='skip delta debugging for spec inference (default: %(default)s)')
    parser.add_argument('--finish-after-fault-localize', action='store_true',
                        help='finish after fault localization (default: %(default)s)')
    parser.add_argument('--version', action='version', version='Angelix 1.1')

    args = parser.parse_args()

    working_dir = join(os.getcwd(), ".angelix")
    if not args.keep_angelix_dir:
        if exists(working_dir):
            shutil.rmtree(working_dir, onerror=rm_force)
        os.mkdir(working_dir)

    rootLogger = logging.getLogger()
    FORMAT = logging.Formatter('%(levelname)-8s %(name)-15s %(message)s')
    if args.quiet:
        rootLogger.setLevel(logging.WARNING)
    elif args.log is not None:
        log_level = getattr(logging, args.log, None)
        rootLogger.setLevel(log_level)
    else:
        rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler("{0}/{1}.log".format(working_dir, 'angelix'))
    fileHandler.setFormatter(FORMAT)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(FORMAT)
    rootLogger.addHandler(consoleHandler)

    if is_subdir(args.src, os.getcwd()):
        logger.error('angelix must be run outside of the source directory')
        exit(1)

    if vars(args)['assert'] is not None and not args.dump_only:
        with open(vars(args)['assert']) as output_file:
            asserts = json.load(output_file)
    else:
        asserts = None

    if 'guards' in args.defect and 'assignments' in args.defect:
        logger.error('\'guards\' and \'assignments\' defect classes are currently incompatible')
        exit(1)

    if args.dump_only:
        if args.golden is not None:
            logger.warning('--dump-only disables --golden option')
        if asserts is not None:
            logger.warning('--dump-only disables --assert option')

    config = dict()
    config['initial_tests']         = args.initial_tests
    config['all_tests']             = args.all_tests
    config['max_z3_trials']         = args.max_z3_trials
    config['defect']                = args.defect
    config['test_timeout']          = args.test_timeout
    config['single_group']          = args.single_group
    config['group_size']            = args.group_size
    config['group_by_score']        = args.group_by_score
    config['localize_from_bottom']  = args.localize_from_bottom
    config['suspicious']            = args.suspicious
    config['localization']          = args.localization
    config['ignore_trivial']        = args.ignore_trivial
    config['path_solving_timeout']  = args.path_solving_timeout
    config['timeout']               = args.timeout
    config['max_angelic_paths']     = args.max_angelic_paths
    config['klee_max_forks']        = args.klee_max_forks
    config['klee_max_depth']        = args.klee_max_depth
    config['klee_search']           = args.klee_search
    config['klee_timeout']          = args.klee_timeout
    config['klee_out_dir_timeout']  = args.klee_out_dir_timeout
    config['klee_solver_timeout']   = args.klee_solver_timeout
    config['klee_debug']            = args.klee_debug
    config['klee_ignore_errors']    = args.klee_ignore_errors
    config['ignore_trans_errors']   = args.ignore_trans_errors
    config['ignore_infer_errors']   = args.ignore_infer_errors
    config['ignore_instrument']     = args.ignore_instrument
    config['ignore_unmatched_execution'] = args.ignore_unmatched_execution
    config['ignore_z3_exception']   = args.ignore_z3_exception
    config['skip_validating_angelic_path'] = args.skip_validating_angelic_path
    config['use_nsynth']            = args.use_nsynth
    config['use_osynth']            = args.use_osynth
    config['use_gcc']               = args.use_gcc
    config['use_frontend_for_test'] = args.use_frontend_for_test
    config['keep_positive_behavior'] = args.keep_positive_behavior
    config['synthesis_timeout']     = args.synthesis_timeout
    config['synthesis_levels']      = args.synthesis_levels
    config['synthesis_global_vars'] = args.synthesis_global_vars
    config['synthesis_func_params'] = args.synthesis_func_params
    config['synthesis_used_vars']   = True  # for backward compatibility
    config['synthesis_ptr_vars']    = args.synthesis_ptr_vars
    config['synthesis_bool_only']   = args.synthesis_bool_only
    config['forced_to_use_bool']    = args.forced_to_use_bool
    config['empty_env_exps']        = args.empty_env_exps
    config['exclude_member_exp']    = args.exclude_member_exp
    config['generate_all']          = args.generate_all
    config['init_uninit_vars']      = args.init_uninit_vars
    config['verbose']               = args.verbose
    config['build_before_instr']    = args.build_before_instr
    config['instr_printf']          = args.instr_printf
    config['mute_config_message']   = args.mute_config_message
    config['mute_build_message']    = args.mute_build_message
    config['mute_test_message']     = args.mute_test_message
    config['show_test_message']     = args.show_test_message
    config['show_oracle_contents']  = args.show_oracle_contents
    config['show_syn_message']      = args.show_syn_message
    config['mute_warning']          = args.mute_warning
    config['show_suspicious_locations'] = args.show_suspicious_locations
    config['invalid_localization']  = args.invalid_localization
    config['angelic_search_strategy'] = args.angelic_search_strategy
    config['step_method']           = args.step_method
    config['compilation_db_file']   = args.compilation_db_file
    config['keep_angelix_dir']      = args.keep_angelix_dir
    config['skip_configure']        = args.skip_configure
    config['skip_build']            = args.skip_build
    config['gobble_klee_message']   = args.gobble_klee_message
    config['lines']                 = args.lines
    config['search_max_trials']     = args.search_max_trials
    config['max_same_cost_iter']    = args.max_same_cost_iter
    config['mcmc_beta']             = args.mcmc_beta
    config['one_bit_flip_prob']     = args.one_bit_flip_prob
    config['chunk_bits']            = args.chunk_bits
    config['max_bits']              = args.max_bits
    config['max_resample']          = args.max_resample
    config['block_expand_factor']   = args.block_expand_factor
    config['inc_fix']               = args.inc_fix
    config['fixed_bv_len']          = args.fixed_bv_len
    config['default_max_cost']      = args.default_max_cost
    config['error_cost']            = args.error_cost
    config['warning_cost']          = args.warning_cost
    config['max_syn_attempts']      = args.max_syn_attempts
    config['max_repair_attempts']   = args.max_repair_attempts
    config['penalty1']              = args.penalty1
    config['penalty2']              = args.penalty2
    config['parallel_testing']      = args.parallel_testing
    config['all_suspicious']        = args.all_suspicious
    config['mutate']                = int(args.mutate)
    config['epsilon']               = float(args.epsilon)
    config['always_accept']         = args.always_accept
    config['random_cost']           = args.random_cost
    config['max_random_cost']       = float(args.max_random_cost)
    config['skip_dd']               = args.skip_dd
    config['spec_from_only_negative'] = args.spec_from_only_negative
    config['finish_after_fault_localize'] = args.finish_after_fault_localize
    config['default_susp_score']    = args.default_susp_score
    config['additional_susp_locs']  = args.additional_susp_locs
    config['default_non_zero_cost'] = args.default_non_zero_cost
    config['timeout_cost']          = args.timeout_cost

    logger.debug('tests: {}'.format(args.tests))
    if args.verbose:
        logger.info('arg oracle = {}'.format(args.oracle))
        for key, value in config.items():
            logger.info('option {} = {}'.format(key, value))

    statistics.init(working_dir, config)

    if args.ignore_lines:
        args.lines = None

    tool = Angelix(working_dir,
                   src=args.src,
                   buggy=args.buggy,
                   oracle=abspath(args.oracle),
                   tests=args.tests,
                   golden=args.golden,
                   asserts=asserts,
                   lines=args.lines,
                   build=args.build,
                   configure=args.configure,
                   config=config)

    if args.dump_only:
        try:
            dump = tool.dump_outputs()
            with open('dump.json', 'w') as output_file:
                asserts = json.dump(dump, output_file, indent=2)
            logger.info('outputs successfully dumped (see dump.json)')
            exit(0)
        except (CompilationError, TransformationError):
            logger.info('failed to dump outputs')
            exit(1)

    if args.tests_summary:
        # run validation tests
        positive, negative = tool.evaluate_ts(tool.validation_src, args.tests)
        val_test_result = {'positive': sorted(positive), 'negative': sorted(negative)}
        logger.info('positive: {}, negative: {}'.format(positive, negative))

        # run golden tests
        src = tool.golden_src
        src.configure()
        src.build()
        positive, negative = tool.evaluate_ts(src, args.tests)
        golden_test_result = {'positive': sorted(positive), 'negative': sorted(negative)}
        logger.info('positive: {}, negative: {}'.format(positive, negative))

        # delta
        negative = list(set(val_test_result['negative']).difference(set(golden_test_result['negative'])))
        delta = {'positive': val_test_result['positive'], 'negative': sorted(negative)}
        logger.info('positive: {}, negative: {}'.format(delta['positive'], delta['negative']))

        summary = {'validation': val_test_result,
                   'golden': golden_test_result,
                   'delta': delta }

        summary_file = join(working_dir, "tests-summary.json")
        with open(summary_file, "w") as write_file:
            json.dump(summary, write_file, indent=4)
        exit(0)

    if config['mutate'] > 0:
        mutate(config, args.buggy, working_dir, args.golden,
               args.build, args.configure)
        exit(0)

    logger.debug('start to measure time')
    start = time.time()

    try:
        if args.timeout is not None:
            with time_limit(args.timeout):
                patches = repair()
        else:
            patches = repair()
    except TimeoutException:
        logger.info("failed to generate patch (timeout)")
        print('TIMEOUT')

        statistics.data['patch_found'] = False
        statistics.data['timeout_occurred'] = True
        statistics.data['time']['total'] = args.timeout
        statistics.save()
        exit(0)
    except (CompilationError, InferenceError, TransformationError):
        logger.info("failed to generate patch")
        print('FAIL')

        statistics.data['patch_found'] = False
        statistics.save()

        exit(1)

    end = time.time()
    elapsed = format_time(end - start)
    statistics.data['time']['total'] = end - start
    statistics.save()

    if not patches:
        logger.info("no patch generated in {}".format(elapsed))
        print('FAIL')

        statistics.data['patch_found'] = False
        statistics.save()

        exit(0)
    else:
        if config['generate_all']:
            patch_dir = basename(abspath(args.src)) + '-' + time.strftime("%Y-%b%d-%H%M%S")
            if not exists(patch_dir):
                os.mkdir(patch_dir)
            for idx, patch in enumerate(patches):
                patch_file = os.path.join(patch_dir, str(idx) + '.patch')
                with open(patch_file, 'w+') as file:
                    for line in patch:
                        file.write(line)
            logger.info("patches successfully generated in {} (see {})".format(elapsed, patch_dir))
        else:
            patch_file = basename(abspath(args.src)) + '-' + time.strftime("%Y-%b%d-%H%M%S") + '.patch'
            logger.info("patch successfully generated in {} (see {})".format(elapsed, patch_file))
            with open(patch_file, 'w+') as file:
                for line in patches[0]:
                    file.write(line)
        print('SUCCESS')

        statistics.data['src'] = args.src
        statistics.data['buggy'] = args.buggy
        statistics.data['patch_found'] = True
        statistics.data['patch_file'] = patch_file
        statistics.save()

        exit(0)
