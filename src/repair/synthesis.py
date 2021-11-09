import sys
import os
import subprocess
from utils import cd
import logging
import json
from pprint import pprint
import tempfile
import shutil
from os.path import join
import statistics
import time
from os import mkdir


logger = logging.getLogger(__name__)


class Synthesizer:

    def __init__(self, config, extracted, syn_dir, last_af_file):
        self.config = config
        self.extracted = extracted
        self.syn_dir = syn_dir
        self.last_af_file = last_af_file
        self.syn_id = 0

    def dump_angelic_forest(self, angelic_forest, af_file):
        '''
        Convert angelic forest to format more suitable for current synthesis engine
        '''
        def id(expr):
            return '{}-{}-{}-{}'.format(*expr)

        dumpable_angelic_forest = dict()
        for test, paths in angelic_forest.items():
            dumpable_paths = []
            for path in paths:
                dumpable_path = []

                for expr, values in path.items():
                    for instance, value in enumerate(values):
                        angelic, _, environment = value  # ignore original for now
                        context = []
                        for name, value in environment.items():
                            context.append({'name': name,
                                            'value': value})
                        dumpable_path.append({ 'context': context,
                                               'value': { 'name': 'angelic',
                                                          'value': angelic },
                                               'expression': id(expr),
                                               'instId': instance })
                dumpable_paths.append(dumpable_path)
            dumpable_angelic_forest[test] = dumpable_paths

        with open(af_file, 'w') as file:
            json.dump(dumpable_angelic_forest, file, indent=2)
        shutil.copyfile(af_file, self.last_af_file)

    def __call__(self, angelic_forest):
        org_syn_timeout = self.config['synthesis_timeout']
        repeat = 0
        while True:
            fix = self.synthesize(angelic_forest)
            if fix is not None:
                self.config['synthesis_timeout'] = org_syn_timeout
                return fix
            elif repeat >= self.config['max_syn_attempts'] - 1:
                self.config['synthesis_timeout'] = org_syn_timeout
                return None
            else:
                logger.info('retry synthesis')
                self.config['synthesis_timeout'] *= 2
                repeat += 1

    def synthesize(self, angelic_forest):
        self.syn_id += 1
        dirpath = self.syn_dir[self.syn_id]
        patch_file = join(dirpath, 'patch')
        config_file = join(dirpath, 'config.json')
        af_file = join(dirpath, 'angelic-forest.json')

        if type(angelic_forest) == str:
            # angelic_forest is a file
            shutil.copyfile(angelic_forest, af_file)
        else:
            # angelic_forest is a data structure
            self.dump_angelic_forest(angelic_forest, af_file)

        for level in self.config['synthesis_levels']:

            logger.info('synthesizing patch with component level \'{}\''.format(level))

            config = {
                "encodingConfig": {
                    "componentsMultipleOccurrences": True,
                    # better if false, if not enough primitive components, synthesis can fail
                    "phantomComponents": True,
                    "repairBooleanConst": False,
                    "repairIntegerConst": False,
                    "level": "linear"
                },
                "simplification": False,
                "reuseStructure": True if self.config['defect'] != ['ptr-assignments'] else False,
                "spaceReduction": True,
                "componentLevel": level,
                "solverBound": 3,
                "solverTimeout": self.config['synthesis_timeout']
            }

            with open(config_file, 'w') as file:
                json.dump(config, file)

            if self.config['use_nsynth'] and not self.config['use_osynth']:
                logger.info('use nsynth')
                jar = os.environ['NSYNTH_JAR']
            else:
                logger.info('use old synth')
                jar = os.environ['SYNTHESIS_JAR']

            if self.config['verbose'] or self.config['show_syn_message']:
                stderr = None
            else:
                stderr = subprocess.DEVNULL

            args = [af_file, self.extracted, patch_file, config_file]

            synthesis_start_time = time.time()

            try:
                logger.debug('call synthesizer: java -jar {} {}'.format(jar, args))
                result = subprocess.check_output(['java', '-jar', jar] + args, stderr=stderr)
            except subprocess.CalledProcessError:
                logger.warning("synthesis returned non-zero code")
                continue
            finally:
                synthesis_end_time = time.time()
                synthesis_elapsed = synthesis_end_time - synthesis_start_time
                statistics.data['time']['synthesis'] += synthesis_elapsed
                iter_stat = dict()
                iter_stat['tests'] = len(angelic_forest)
                iter_stat['level'] = level
                iter_stat['time'] = synthesis_elapsed
                statistics.data['iterations']['synthesis'].append(iter_stat)
                statistics.save()

            result_stripped = str(result, 'UTF-8').strip()
            logger.debug('synthesis output: {}'.format(result_stripped))
            if 'TIMEOUT' in result_stripped:
                logger.warning('timeout when synthesizing fix')
            elif 'FAIL' in result_stripped:
                logger.info('synthesis failed')
            elif 'SUCCESS' in result_stripped:
                with open(patch_file) as file:
                    content = file.readlines()
                original = None
                patch = dict()
                while len(content) > 0:
                    line = content.pop(0)
                    if len(line) == 0:
                        continue
                    expr = tuple(map(int, line.strip().split('-')))

                    def convert_to_c(s):
                        return s.replace('_LBRSQR_', '[').replace('_RBRSQR_', ']')

                    original = convert_to_c(content.pop(0).strip())
                    fixed = convert_to_c(content.pop(0).strip())
                    logger.info('fixing expression {}: {} ---> {}'.format(expr, original, fixed))
                    patch[expr] = fixed
                if len(patch) == 0:
                    logger.warn('patch contains no changes')
                    return None, level, original
                return patch, level, original
            else:
                raise Exception('result: ' + str(result, 'UTF-8'))

        return None, None, None
