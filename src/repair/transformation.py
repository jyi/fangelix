import os
import sys
from os.path import join, basename, relpath
import tempfile
import subprocess
from utils import cd
import logging
import shutil


logger = logging.getLogger(__name__)


class TransformationError(Exception):
    pass


class RepairableTransformer:

    def __init__(self, config, extracted):
        self.config = config
        if self.config['verbose']:
            self.subproc_output = sys.stderr
        else:
            self.subproc_output = subprocess.DEVNULL
        self.extracted = extracted

    def __call__(self, project):
        src = basename(project.dir)
        logger.info('instrumenting repairable of {} source'.format(src))
        environment = dict(os.environ)
        if 'if-conditions' in self.config['defect']:
            environment['ANGELIX_IF_CONDITIONS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_IF_CONDITIONS_DEFECT_CLASS=YES')
        if 'assignments' in self.config['defect']:
            environment['ANGELIX_ASSIGNMENTS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_ASSIGNMENTS_DEFECT_CLASS=YES')
        if 'ptr-assignments' in self.config['defect']:
            environment['ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS=YES')
        if 'loop-conditions' in self.config['defect']:
            environment['ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS=YES')
        if 'deletions' in self.config['defect']:
            environment['ANGELIX_DELETIONS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_DELETIONS_DEFECT_CLASS=YES')
        if 'guards' in self.config['defect']:
            environment['ANGELIX_GUARDS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_GUARDS_DEFECT_CLASS=YES')
        if self.config['ignore_trivial']:
            environment['ANGELIX_IGNORE_TRIVIAL'] = 'YES'
            logger.debug('ANGELIX_IGNORE_TRIVIAL=YES')
        if self.config['angelic_search_strategy'] != 'symbolic':
            environment['ANGELIX_EXTRACTED'] = self.extracted
            logger.debug('ANGELIX_EXTRACTED={}'.format(self.extracted))

        if self.config['synthesis_global_vars']:
            environment['ANGELIX_GLOBAL_VARIABLES'] = 'YES'
            logger.debug('ANGELIX_GLOBAL_VARIABLES=YES')

        if self.config['synthesis_func_params']:
            environment['ANGELIX_FUNCTION_PARAMETERS'] = 'YES'
            logger.debug('ANGELIX_FUNCTION_PARAMETERS=YES')

        if self.config['synthesis_used_vars']:
            environment['ANGELIX_USED_VARIABLES'] = 'YES'
            logger.debug('ANGELIX_USED_VARIABLES=YES')

        if self.config['forced_to_use_bool']:
            environment['FORCED_TO_USE_BOOL'] = 'YES'
            logger.debug('FORCED_TO_USE_BOOL=YES')

        if self.config['synthesis_ptr_vars']:
            environment['ANGELIX_POINTER_VARIABLES'] = 'YES'
            logger.debug('ANGELIX_POINTER_VARIABLES=YES')

        if self.config['init_uninit_vars']:
            environment['ANGELIX_INIT_UNINIT_VARS'] = 'YES'
            logger.debug('ANGELIX_INIT_UNINIT_VARS=YES')

        if self.config['empty_env_exps']:
            environment['ANGELIX_EMPTY_ENV_EXPS'] = 'YES'
            logger.debug('ANGELIX_EMPTY_ENV_EXPS=YES')

        if self.config['exclude_member_exp']:
            environment['ANGELIX_EXLUCDE_MEMBER_EXPR'] = 'YES'
            logger.debug('ANGELIX_EXLUCDE_MEMBER_EXPR=YES')

        with cd(project.dir):
            logger.debug('trans dir: {}'.format(project.dir))
            logger.debug('trans cmd: instrument-repairable {}'.format(project.buggy))
            return_code = subprocess.call(['instrument-repairable', project.buggy],
                                          stderr=self.subproc_output,
                                          stdout=self.subproc_output,
                                          env=environment)
            shutil.copyfile(project.buggy, project.buggy + '.trans')
        if return_code != 0:
            if self.config['ignore_trans_errors']:
                logger.warning("transformation of {} failed".format(relpath(project.dir)))
            else:
                logger.error("transformation of {} failed".format(relpath(project.dir)))
                raise TransformationError()


class SuspiciousTransformer:

    def __init__(self, config, extracted):
        self.config = config
        self.extracted = extracted
        if self.config['verbose']:
            self.subproc_output = sys.stderr
        else:
            self.subproc_output = subprocess.DEVNULL

    def __call__(self, project, locations):
        locIdx = 1
        src = basename(project.dir)
        logger.info('instrumenting suspicious of {} source'.format(src))
        environment = dict(os.environ)
        dirpath = tempfile.mkdtemp()
        suspicious_file = join(dirpath, 'suspicious')
        logger.debug('suspicious_file: {}'.format(suspicious_file))
        logger.debug('locations: {}'.format(locations))
        with open(suspicious_file, 'w') as file:
            for loc in locations:
                file.write('{} {} {} {}\n'.format(*loc[locIdx]))

        environment['ANGELIX_EXTRACTED'] = self.extracted
        environment['ANGELIX_SUSPICIOUS'] = suspicious_file

        logger.debug('ANGELIX_EXTRACTED={}'.format(self.extracted))
        logger.debug('ANGELIX_SUSPICIOUS={}'.format(suspicious_file))

        if 'if-conditions' in self.config['defect']:
            environment['ANGELIX_IF_CONDITIONS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_IF_CONDITIONS_DEFECT_CLASS=YES')
        if 'assignments' in self.config['defect']:
            environment['ANGELIX_ASSIGNMENTS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_ASSIGNMENTS_DEFECT_CLASS=YES')
        if 'ptr-assignments' in self.config['defect']:
            environment['ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS=YES')
        if 'loop-conditions' in self.config['defect']:
            environment['ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS=YES')
        if 'deletions' in self.config['defect']:
            environment['ANGELIX_DELETIONS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_DELETIONS_DEFECT_CLASS=YES')
        if 'guards' in self.config['defect']:
            environment['ANGELIX_GUARDS_DEFECT_CLASS'] = 'YES'
            logger.debug('ANGELIX_GUARDS_DEFECT_CLASS=YES')
        if self.config['ignore_trivial']:
            environment['ANGELIX_IGNORE_TRIVIAL'] = 'YES'
            logger.debug('ANGELIX_IGNORE_TRIVIAL=YES')

        if self.config['synthesis_global_vars']:
            environment['ANGELIX_GLOBAL_VARIABLES'] = 'YES'

        if self.config['synthesis_func_params']:
            environment['ANGELIX_FUNCTION_PARAMETERS'] = 'YES'

        if self.config['synthesis_used_vars']:
            environment['ANGELIX_USED_VARIABLES'] = 'YES'

        if self.config['synthesis_ptr_vars']:
            environment['ANGELIX_POINTER_VARIABLES'] = 'YES'

        if self.config['forced_to_use_bool']:
            environment['FORCED_TO_USE_BOOL'] = 'YES'
            logger.debug('FORCED_TO_USE_BOOL=YES')

        if self.config['init_uninit_vars']:
            environment['ANGELIX_INIT_UNINIT_VARS'] = 'YES'

        if self.config['empty_env_exps']:
            environment['ANGELIX_EMPTY_ENV_EXPS'] = 'YES'
            logger.debug('ANGELIX_EMPTY_ENV_EXPS=YES')

        if self.config['exclude_member_exp']:
            environment['ANGELIX_EXLUCDE_MEMBER_EXPR'] = 'YES'
            logger.debug('ANGELIX_EXLUCDE_MEMBER_EXPR=YES')

        with cd(project.dir):
            logger.debug('trans dir: {}'.format(project.dir))
            logger.debug('trans cmd: instrument-suspicious {}'.format(project.buggy))
            return_code = subprocess.call(['instrument-suspicious', project.buggy],
                                          stderr=self.subproc_output,
                                          stdout=self.subproc_output,
                                          env=environment)
        if return_code != 0:
            if self.config['ignore_trans_errors']:
                logger.warning("transformation of {} failed".format(relpath(project.dir)))
            else:
                logger.error("transformation of {} failed".format(relpath(project.dir)))
                raise TransformationError()

        shutil.rmtree(dirpath)


class FixInjector:

    def __init__(self, config):
        self.config = config
        if self.config['verbose']:
            self.subproc_output = sys.stderr
        else:
            self.subproc_output = subprocess.DEVNULL

    def __call__(self, project, patch):
        src = basename(project.dir)
        logger.info('applying patch to {} source'.format(src))

        environment = dict(os.environ)
        dirpath = tempfile.mkdtemp()
        patch_file = join(dirpath, 'patch')
        with open(patch_file, 'w') as file:
            for e, p in patch.items():
                file.write('{} {} {} {}\n'.format(*e))
                file.write(p + "\n")

        environment['ANGELIX_PATCH'] = patch_file

        with cd(project.dir):
            logger.debug('trans dir: {}'.format(project.dir))
            logger.debug('trans cmd: instrument-suspicious {}'.format(project.buggy))
            return_code = subprocess.call(['apply-patch', project.buggy],
                                          stderr=self.subproc_output,
                                          stdout=self.subproc_output,
                                          env=environment)
        if return_code != 0:
            if self.config['ignore_trans_errors']:
                logger.error("transformation of {} failed".format(relpath(project.dir)))
            else:
                logger.error("transformation of {} failed".format(relpath(project.dir)))
                raise TransformationError()
        shutil.rmtree(dirpath)


class PrintfTransformer:

    def __init__(self, config):
        self.config = config
        if self.config['verbose']:
            self.subproc_output = sys.stderr
        else:
            self.subproc_output = subprocess.DEVNULL

    def __call__(self, project, source_file):
        src = basename(project.dir)
        logger.info('instrumenting printfs of {} source'.format(src))

        with cd(project.dir):
            logger.debug('trans dir: {}'.format(project.dir))
            logger.debug('trans cmd: instrument-suspicious {}'.format(project.buggy))
            return_code = subprocess.call(['instrument-printf', source_file],
                                          stderr=self.subproc_output,
                                          stdout=self.subproc_output)
            with open(source_file, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write('#ifndef ANGELIX_OUTPUT\n#define ANGELIX_OUTPUT(type, expr, id) expr\n#endif\n' + content)

        if return_code != 0:
            if self.config['ignore_trans_errors']:
                logger.error("transformation of {} failed".format(relpath(project.dir)))
            else:
                logger.error("transformation of {} failed".format(relpath(project.dir)))
                raise TransformationError()

        pass


class MutateTransformer:

    def __init__(self, config, extracted):
        self.config = config
        if self.config['verbose']:
            self.subproc_output = sys.stderr
        else:
            self.subproc_output = subprocess.DEVNULL
        self.extracted = extracted

    def __call__(self, project):
        src = basename(project.dir)
        logger.info('instrumenting repairable of {} source'.format(src))
        environment = dict(os.environ)
        if 'if-conditions' in self.config['defect']:
            environment['ANGELIX_IF_CONDITIONS_DEFECT_CLASS'] = 'YES'
        if 'assignments' in self.config['defect']:
            environment['ANGELIX_ASSIGNMENTS_DEFECT_CLASS'] = 'YES'
        if 'loop-conditions' in self.config['defect']:
            environment['ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS'] = 'YES'
        if 'deletions' in self.config['defect']:
            environment['ANGELIX_DELETIONS_DEFECT_CLASS'] = 'YES'
        if 'guards' in self.config['defect']:
            environment['ANGELIX_GUARDS_DEFECT_CLASS'] = 'YES'
        if self.config['ignore_trivial']:
            environment['ANGELIX_IGNORE_TRIVIAL'] = 'YES'
        if self.config['angelic_search_strategy'] != 'symbolic':
            environment['ANGELIX_EXTRACTED'] = self.extracted
        with cd(project.dir):
            logger.debug('trans dir: {}'.format(project.dir))
            logger.debug('trans cmd: instrument-suspicious {}'.format(project.buggy))
            logger.debug('environment for mutate: {}'.format(environment))
            return_code = subprocess.call(['mutate', project.buggy],
                                          stderr=self.subproc_output,
                                          stdout=self.subproc_output,
                                          env=environment)
        if return_code != 0:
            if self.config['ignore_trans_errors']:
                logger.warning("transformation of {} failed".format(relpath(project.dir)))
            else:
                logger.error("transformation of {} failed".format(relpath(project.dir)))
                raise TransformationError()
