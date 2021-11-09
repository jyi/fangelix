import os
from os.path import basename, join, exists
from utils import cd
import subprocess
import logging
import sys
import tempfile
from utils import env_log
from glob import glob
from enum import Enum

logger = logging.getLogger(__name__)


class RunMode(Enum):
    NAITIVE = 1
    KLEE = 2
    KLEE_SP = 3  # KLEE with single path
    ZESTI = 4


class Tester:

    def __init__(self, config, oracle, workdir):
        self.config = config
        self.oracle = oracle
        self.workdir = workdir

    def __call__(self, project, test,
                 dump=None, trace=None, load=None,
                 run_mode=RunMode.NAITIVE,
                 env=os.environ,
                 check_instrumented=False,
                 ignore_timeout=False):
        src = basename(project.dir)
        environment = dict(env)
        if run_mode == RunMode.KLEE:
            logger.info('running test \'{}\' of {} source with KLEE'.format(test, src))
            environment['KLEE_RUN_MODE'] = 'KLEE'
        elif run_mode == RunMode.KLEE_SP:
            logger.info('running test \'{}\' of {} source with KLEE_SP'.format(test, src))
            environment['KLEE_RUN_MODE'] = 'KLEE_SP'
        elif run_mode == RunMode.ZESTI:
            logger.info('running test \'{}\' of {} source with ZESTI'.format(test, src))
            environment['KLEE_RUN_MODE'] = 'ZESTI'
        else:
            logger.info('running test \'{}\' of {} source'.format(test, src))

        if dump is not None:
            environment['ANGELIX_WITH_DUMPING'] = dump
            reachable_dir = join(dump, 'reachable')  # maybe it should be done in other place?
            os.mkdir(reachable_dir)
        if trace is not None:
            environment['ANGELIX_WITH_TRACING'] = trace
        if (trace is not None) or (dump is not None) or (load is not None):
            environment['ANGELIX_RUN'] = 'angelix-run-test'
        if run_mode == RunMode.KLEE or \
           run_mode == RunMode.ZESTI or \
           run_mode == RunMode.KLEE_SP:
            environment['ANGELIX_RUN'] = 'angelix-run-klee'
            # using stub library to make lli work
            environment['LLVMINTERP'] = 'lli -load {}/libkleeRuntest.so'.format(os.environ['KLEE_LIBRARY_PATH'])
        if load is not None:
            environment['ANGELIX_WITH_LOADING'] = load
        environment['ANGELIX_WORKDIR'] = self.workdir
        environment['ANGELIX_TEST_ID'] = test

        dirpath = tempfile.mkdtemp()
        executions = join(dirpath, 'executions')

        environment['ANGELIX_RUN_EXECUTIONS'] = executions

        if self.config['verbose'] and not self.config['mute_test_message']:
            subproc_output = sys.stderr
        else:
            subproc_output = subprocess.DEVNULL

        if self.config['show_oracle_contents']:
            logger.debug('oracle:\n')
            with open(self.oracle) as f:
                for line in f.readlines():
                    logger.debug('{}'.format(line.strip('\n')))

        # logger.debug('environment: {}'.format(environment))
        env_log(logger, 'KLEE_RUN_MODE', environment)
        env_log(logger, 'ANGELIX_WITH_DUMPING', environment)
        env_log(logger, 'ANGELIX_WITH_TRACING', environment)
        env_log(logger, 'ANGELIX_RUN', environment)
        # env_log(logger, 'LLVMINTERP', environment)
        env_log(logger, 'ANGELIX_WITH_LOADING', environment)
        env_log(logger, 'ANGELIX_WORKDIR', environment)
        env_log(logger, 'ANGELIX_TEST_ID', environment)
        # env_log(logger, 'ANGELIX_RUN_EXECUTIONS', environment)
        env_log(logger, 'ANGELIX_SYMBOLIC_RUNTIME', environment)
        env_log(logger, 'ANGELIX_LOAD_JSON', environment)
        env_log(logger, 'ANGELIX_LOAD', environment)
        env_log(logger, 'ANGELIX_TRACE', environment)
        env_log(logger, 'CC', environment)
        env_log(logger, 'GOBLE_KLEE_MESSAGE', environment)

        with cd(project.dir):
            logger.debug('cwd: {}'.format(os.getcwd()))
            logger.debug('run: {} {}'.format(self.oracle, test))
            if self.config['verbose']:
                # Does not work if oracle gobbles the output.
                proc = subprocess.Popen([self.oracle, test],
                                        env=environment,
                                        stdout=subproc_output,
                                        stderr=subproc_output,
                                        shell=False)
            else:
                proc = subprocess.Popen([self.oracle, test],
                                        env=environment,
                                        stdout=subproc_output,
                                        stderr=subproc_output,
                                        shell=False)
            try:
                if run_mode == RunMode.KLEE or \
                   run_mode == RunMode.ZESTI or \
                   self.config['test_timeout'] is None:  # KLEE has its own timeout
                    code = proc.wait()
                else:
                    code = proc.wait(timeout=self.config['test_timeout'])
            except subprocess.TimeoutExpired:
                logger.warning('timeout in test {}'.format(test))
                os.system('for id in $(pgrep klee); do kill -9 $id; done')
                cost_file = environment['ANGELIX_COST_FILE'] \
                    if 'ANGELIX_COST_FILE' in environment else None
                if cost_file is not None:
                    logger.debug('assign timeout cost to {}'.format(cost_file))
                    with open(cost_file, 'w') as file:
                        file.write(self.config['timeout_cost'])
                if ignore_timeout:
                    raise TestTimeout
                else:
                    code = 1

            logger.info('output code: {}'.format(code))

        instrumented = True
        if dump is not None or trace is not None or \
           run_mode == RunMode.KLEE or \
           run_mode == RunMode.ZESTI or \
           run_mode == RunMode.KLEE_SP:
            if exists(executions):
                with open(executions) as file:
                    content = file.read()
                    if len(content) > 1:
                        logger.warning("ANGELIX_RUN is executed multiple times by test {}".format(test))
                        # instrumented = False
            else:
                if not self.config['mute_test_message']:
                    logger.warning("ANGELIX_RUN is not executed by test {}".format(test))
                    instrumented = False

        if os.path.exists(executions):
            os.remove(executions)

        if check_instrumented:
            return (code == 0, instrumented)
        else:
            return code == 0


class TestTimeout(Exception):
    pass
