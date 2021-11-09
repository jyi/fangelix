import copy
import difflib
import os
from os.path import join, exists, relpath, basename, realpath
import shutil
import subprocess
import json
from utils import cd
import logging
import tempfile
import sys
import re
import statistics
import time
from transformation import PrintfTransformer
from utils import env_log


logger = logging.getLogger(__name__)


class CompilationError(Exception):
    pass


class Project:

    def __init__(self, config, dir, buggy, build_cmd, configure_cmd):
        self.config = config
        if self.config['verbose']:
            self.subproc_output = sys.stderr
        else:
            self.subproc_output = subprocess.DEVNULL
        self.dir = dir
        self.buggy = buggy
        self.build_cmd = build_cmd
        self.configure_cmd = configure_cmd

    def initialize(self):
        if self.config['instr_printf'] is not None:
            self.configure()
            self.instrument_printf = PrintfTransformer(self.config)
            self.instrument_printf(self, self.config['instr_printf'])

        self._buggy_backup = join(self.dir, self.buggy) + '.backup'
        shutil.copyfile(join(self.dir, self.buggy), self._buggy_backup)

    def restore_buggy(self):
        shutil.copyfile(self._buggy_backup, join(self.dir, self.buggy))

    def diff_buggy(self):
        with open(join(self.dir, self.buggy), encoding='latin-1') as buggy:
            buggy_lines = buggy.readlines()
        with open(self._buggy_backup, encoding='latin-1') as backup:
            backup_lines = backup.readlines()
        return difflib.unified_diff(backup_lines, buggy_lines,
                                    fromfile=join('a', self.buggy),
                                    tofile=join('b', self.buggy))

    def import_compilation_db(self, compilation_db):
        compilation_db = copy.deepcopy(compilation_db)
        for item in compilation_db:
            item['directory'] = join(self.dir, item['directory'])
            item['file'] = join(self.dir, item['file'])
            # this is a temporary hack. It general case, we need (probably) a different workflow:
            wrong_dir = realpath(join(self.dir, '..', 'validation'))
            item['command'] = item['command'].replace(wrong_dir, self.dir)

            item['command'] = item['command'] + ' -I' + os.environ['LLVM3_INCLUDE_PATH']
            # this is a hack to skip output expressions when perform transformation:
            item['command'] = item['command'] + ' -include ' + os.environ['ANGELIX_RUNTIME_H']
            item['command'] = item['command'] + ' -D ANGELIX_INSTRUMENTATION'
        compilation_db_file = join(self.dir, 'compile_commands.json')
        with open(compilation_db_file, 'w') as file:
            json.dump(compilation_db, file, indent=2)

    def configure(self):
        if self.configure_cmd is None or self.config['skip_configure']:
            return

        src = basename(self.dir)
        logger.info('configuring {} source'.format(src))
        logger.debug('configure_cmd: {}'.format(self.configure_cmd))
        if self.config['mute_config_message']:
            cofig_subproc_output = subprocess.DEVNULL
        else:
            cofig_subproc_output = self.subproc_output
        compile_start_time = time.time()
        with cd(self.dir):
            return_code = subprocess.call(self.configure_cmd,
                                          shell=True,
                                          stderr=cofig_subproc_output,
                                          stdout=cofig_subproc_output)
        if return_code != 0 and not self.config['mute_warning']:
            logger.warning("configuration of {} returned non-zero code".format(relpath(self.dir)))
            exit(1)
        compile_end_time = time.time()
        compile_elapsed = compile_end_time - compile_start_time
        statistics.data['time']['compilation'] += compile_elapsed


def build_in_env(dir, cmd, subproc_output, config, env=os.environ):
    dirpath = tempfile.mkdtemp()
    messages = join(dirpath, 'messages')

    environment = dict(env)
    environment['ANGELIX_COMPILER_MESSAGES'] = messages

    env_log(logger, 'CC', env)
    # env_log(logger, 'ANGELIX_COMPILER_CUSTOM_KLEE_LINK', env)
    logger.debug('cmd: {}'.format(cmd))
    with cd(dir):
        return_code = subprocess.Popen(cmd,
                                       env=environment,
                                       shell=True,
                                       stderr=subproc_output,
                                       stdout=subproc_output).wait()

    if return_code != 0 and not config['mute_warning']:
        logger.warning("compilation of {} returned non-zero code".format(relpath(dir)))
        exit(1)

    if exists(messages):
        with open(messages) as file:
            lines = file.readlines()
        if not config['mute_warning']:
            for line in lines:
                logger.warning("failed to build {}".format(relpath(line.strip())))
                exit(1)


def build_with_cc(dir, cmd, subproc_output, cc, config, ar=None, ranlib=None):
    env = dict(os.environ)
    env['CC'] = cc
    if ar is not None:
        env['AR'] = ar
    if ranlib is not None:
        env['RANLIB'] = ranlib
    build_in_env(dir, cmd, subproc_output, config, env)


class Validation(Project):

    def build(self):
        if self.config['skip_build']:
            return

        logger.info('building {} source'.format(basename(self.dir)))
        compile_start_time = time.time()
        build_with_cc(self.dir,
                      self.build_cmd,
                      subprocess.DEVNULL if self.config['mute_build_message']
                      else self.subproc_output,
                      'angelix-compiler --test' if self.config['use_gcc']
                      else 'angelix-compiler --klee',
                      self.config)
        compile_end_time = time.time()
        compile_elapsed = compile_end_time - compile_start_time
        statistics.data['time']['compilation'] += compile_elapsed

    def export_compilation_db(self):
        if not self.config['compilation_db_file']:
            logger.info('building json compilation database from {} source'.format(
                basename(self.dir)))
            compile_start_time = time.time()
            build_with_cc(self.dir,
                          'bear ' + self.build_cmd,
                          subprocess.DEVNULL if self.config['mute_build_message']
                          else self.subproc_output,
                          'angelix-compiler --test' if self.config['use_gcc']
                          else 'angelix-compiler --klee',
                          self.config)
            compile_end_time = time.time()
            compile_elapsed = compile_end_time - compile_start_time
            statistics.data['time']['compilation'] += compile_elapsed
            compilation_db_file = join(self.dir, 'compile_commands.json')
        else:
            compilation_db_file = self.config['compilation_db_file']

        with open(compilation_db_file) as file:
            compilation_db = json.load(file)
        # making paths relative:
        for item in compilation_db:
            item['directory'] = relpath(item['directory'], self.dir)
            item['file'] = relpath(item['file'], self.dir)
        return compilation_db


class Frontend(Project):

    def build(self):
        if self.config['skip_build']:
            return

        if self.config['use_gcc'] and 'assignments' not in self.config['defect']:
            angelix_compiler = 'angelix-compiler --frontend'
        else:
            angelix_compiler = 'angelix-compiler --klee'
        logger.info('building {} source'.format(basename(self.dir)))
        compile_start_time = time.time()
        build_with_cc(self.dir,
                      self.build_cmd,
                      subprocess.DEVNULL if self.config['mute_build_message']
                      else self.subproc_output,
                      angelix_compiler,
                      self.config)
        compile_end_time = time.time()
        compile_elapsed = compile_end_time - compile_start_time
        statistics.data['time']['compilation'] += compile_elapsed


class Golden(Project):

    def build(self):
        if self.config['skip_build']:
            return

        if self.config['use_gcc'] and 'assignments' not in self.config['defect']:
            angelix_compiler = 'angelix-compiler --test'
        else:
            angelix_compiler = 'angelix-compiler --klee'
        logger.info('building {} source'.format(basename(self.dir)))
        compile_start_time = time.time()
        build_with_cc(self.dir,
                      self.build_cmd,
                      subprocess.DEVNULL if self.config['mute_build_message']
                      else self.subproc_output,
                      angelix_compiler,
                      self.config)
        compile_end_time = time.time()
        compile_elapsed = compile_end_time - compile_start_time
        statistics.data['time']['compilation'] += compile_elapsed


class Backend(Project):

    def build(self):
        if self.config['skip_build']:
            return

        logger.info('building {} source'.format(basename(self.dir)))
        compile_start_time = time.time()
        build_with_cc(self.dir,
                      self.build_cmd,
                      subprocess.DEVNULL if self.config['mute_build_message']
                      else self.subproc_output,
                      'angelix-compiler --klee',
                      self.config)
        compile_end_time = time.time()
        compile_elapsed = compile_end_time - compile_start_time
        statistics.data['time']['compilation'] += compile_elapsed
