import logging
import os
from os import mkdir
import json
from functools import reduce
from os.path import join, dirname, relpath, basename
import subprocess
import random
import time
import statistics

logger = logging.getLogger('rand_infer')


class RandomInferrer:

    def __init__(self, config, tester, searchDir, extracted, working_dir):
        self.config = config
        self.run_test = tester
        self.searchDir = searchDir
        self.extracted = extracted
        self.working_dir = working_dir

    def __call__(self, project, test, locations):
        logger.info('inferring specification for test \'{}\' through random search'.format(test))

        self.max_len_dict = dict()
        environment = dict(os.environ)
        spec_found = False
        instance = 0
        explored = 0
        sampled = 0
        sample_space_exhausted = False
        trials_exhuasted = False
        trials_set = set()

        inference_start_time = time.time()
        while instance < self.config['search_max_trials']:
            guess_config, guess_file, trace_file = self.trial(test, locations, instance)
            logger.debug('guess_config: {}'.format(guess_config))
            sampled += 1
            if self.is_sample_space_exhausted(guess_config, trials_set):
                logger.info('sample space exhausted')
                sample_space_exhausted = True
                break

            if guess_config in trials_set:
                logger.info('already tried: {}'.format(guess_config))
                instance += 1
                continue

            trials_set.add(guess_config)
            environment['ANGELIX_LOAD_JSON'] = guess_file
            environment['ANGELIX_TRACE_AFTER_LOAD'] = trace_file

            logger.info('trial #{}'.format(instance))
            # call testing.py: Tester.__call__
            code = self.run_test(project, test, env=environment)
            explored += 1
            if code:
                spec_found = True
                break
            instance += 1

        if spec_found:
            angelic_paths = self.transform_to_angelic_value(trace_file)
        else:
            angelic_paths = []
        trials_exhuasted = not spec_found and instance >= self.config['search_max_trials']

        inference_end_time = time.time()
        inference_elapsed = inference_end_time - inference_start_time
        statistics.data['time']['inference'] += inference_elapsed

        iter_stat = dict()
        iter_stat['locations'] = locations
        iter_stat['test'] = test
        iter_stat['time'] = dict()
        iter_stat['paths'] = dict()
        iter_stat['paths']['explored'] = explored
        iter_stat['paths']['sampled'] = sampled
        iter_stat['paths']['angelic_found'] = spec_found
        iter_stat['paths']['angelic'] = len(angelic_paths)
        iter_stat['paths']['sample_space_exhausted'] = sample_space_exhausted
        iter_stat['paths']['trials_exhuasted'] = trials_exhuasted
        statistics.data['iterations']['random'].append(iter_stat)

        statistics.save()

        return angelic_paths

    def is_sample_space_exhausted(self, guess_config, trials_set):
        total = 0
        for item in guess_config:
            logger.debug('item: {}'.format(item))
            if len(item) != 0:
                total += 2 ** len(item)
            elif len(item) == 0:
                # at runtime, this condition is not executed, but
                # we include in the sample space
                total += 2
        return len(trials_set) >= total

    def gen_guess_random(self, length):
        guess = []
        for idx in range(length):
            guess.append(random.randint(0, 1))
        logger.info('guess generated: {}'.format(guess))
        return guess

    def trial(self, test, expressions, instance):
        assert instance >= 0
        prev_trace_file = join(self.working_dir, "trace", test) if instance == 0 \
                          else join(self.searchDir[test], 'trace', 'trace' + str(instance - 1))
        logger.info('prev_trace_file: {}'.format(prev_trace_file))
        guess_dict = dict()
        for exp in expressions:
            pattern = '\"^' + reduce((lambda x, y: '{} {}'.format(x, y)), exp) + ',\"' \
                      if instance == 0 else \
                        '\"^' + reduce((lambda x, y: '{}-{}'.format(x, y)), exp) + ',\"'
            num_of_exe = int(subprocess.check_output('grep ' + pattern + ' ' + prev_trace_file
                                                     + '| wc -l',
                                                     shell=True).decode('ascii'))

            key = reduce((lambda x, y: '{}-{}'.format(x, y)), exp)
            if self.max_len_dict.get(key) is None:
                self.max_len_dict[key] = num_of_exe
            else:
                self.max_len_dict[key] = max(self.max_len_dict[key], num_of_exe)
            guess_dict[key] = self.gen_guess_random(self.max_len_dict[key])

        if not os.path.exists(self.searchDir[test]):
            mkdir(self.searchDir[test])

        guess_dir = join(self.searchDir[test], 'guess')
        if not os.path.exists(guess_dir):
            mkdir(guess_dir)
        guess_file = join(guess_dir, 'guess' + str(instance) + '.json')

        trace_dir = join(self.searchDir[test], 'trace')
        if not os.path.exists(trace_dir):
            mkdir(trace_dir)
        cur_trace_file = join(trace_dir, 'trace' + str(instance))

        with open(guess_file, 'w') as file:
            file.write(json.dumps(guess_dict))
        logger.debug('guess_dict: {}'.format(guess_dict))
        guess_config = tuple([tuple(guess_dict[key]) for key in guess_dict.keys()])
        return guess_config, guess_file, cur_trace_file

    '''
    ctxt: e.g., n = 2 ; x = 1
    return: {'n': 2, 'x': 1}
    '''
    def parseCtxt(self, ctxt):
        logger.debug('ctxt: {}'.format(ctxt))

        def parseAssignment(a):
            var, val = list(map(lambda x: x.strip(), a.split("=")))
            return {var: int(val)}

        assignments = list(map(lambda x: x.strip(), ctxt.split(';')))
        dics = list(map(lambda x: parseAssignment(x), assignments))
        return reduce(lambda x, y: {**x, **y}, dics)

    '''
    loc: e.g., 10-10-10-14
    return (10, 10, 10, 14)
    '''
    def parseLoc(self, loc):
        l1, l2, l3, l4 = loc.split('-', maxsplit=4)
        return (int(l1), int(l2), int(l3), int(l4))

    def transform_to_angelic_value(self, trace_file):
        specDic = dict()
        logger.debug('trace_file: {}'.format(trace_file))
        with open(trace_file) as f:
            for _, line in enumerate(f):
                loc, angelic, ctxt = line.split(", ", maxsplit=3)
                pLoc = self.parseLoc(loc)
                if specDic.get(pLoc) is None:
                    specDic[pLoc] = [(True if int(angelic) == 1 else False,
                                      None,
                                      self.parseCtxt(ctxt))]
                else:
                    specDic[pLoc].append((True if int(angelic) == 1 else False,
                                          None,
                                          self.parseCtxt(ctxt)))

        return [specDic]
