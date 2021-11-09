from os.path import join, relpath, basename
from utils import NoSmtError, InferenceError, z3util
from utils import LocationInfo as LI
import logging
from glob import glob
import os
import time
import statistics
import shutil
import threading
from watchdog.events import FileSystemEventHandler
import z3
from z3 import Select, Concat, Array, BitVecSort, BitVecVal, Solver, BitVec
from watchdog.observers import Observer
from testing import RunMode
import z3types


logger = logging.getLogger('sym_infer')
klee_out_available = threading.Event()


class Smt2FileParseException(Exception):
    pass


class Smt2VarParseException(Exception):
    pass


class UnmatchedExecution(Exception):
    pass


def parse_variables(vars):
    '''
    <type> ! choice ! <line> ! <column> ! <line> ! <column> ! <instance> ! angelic
    <type> ! choice ! <line> ! <column> ! <line> ! <column> ! <instance> ! original
    <type> ! choice ! <line> ! <column> ! <line> ! <column> ! <instance> ! env ! <name>
    <type> ! const ! <line> ! <column> ! <line> ! <column>
    <type> ! output ! <name> ! <instance>
    reachable ! <name> ! <instance>

    returns outputs, choices, constants, reachable, original

    outputs: name -> type * num of instances
    choices: expr -> type * num of instances * env-name list
    constants: expr list
    reachable: set of strings
    original: if original available

    Note: assume environment variables are always int
    '''
    output_type = dict()
    output_instances = dict()
    choice_type = dict()
    choice_instances = dict()
    choice_env = dict()
    reachable = set()
    constants = set()
    original = False
    for v in vars:
        tokens = v.split('!')
        first = tokens.pop(0)
        if first == 'reachable':
            label = tokens.pop(0)
            reachable.add(label)
        else:
            type = first
            kind = tokens.pop(0)
            if kind == 'output':
                name, instance = tokens.pop(0), int(tokens.pop(0))
                output_type[name] = type
                if name not in output_instances:
                    output_instances[name] = []
                output_instances[name].append(instance)
            elif kind == 'choice':
                expr = int(tokens.pop(0)), int(tokens.pop(0)), int(tokens.pop(0)), int(tokens.pop(0))
                instance = int(tokens.pop(0))
                value = tokens.pop(0)
                if value == 'angelic':
                    choice_type[expr] = type
                    if expr not in choice_env: # because it can be empty
                        choice_env[expr] = set()
                    if expr not in choice_instances:
                        choice_instances[expr] = []
                    choice_instances[expr].append(instance)
                elif value == 'original':
                    original = True
                elif value == 'env':
                    name = tokens.pop(0)
                    if expr not in choice_env:
                        choice_env[expr] = set()
                    choice_env[expr].add(name)
                else:
                    raise InferenceError()
            elif kind == 'const':
                logger.error('constant choices are not supported')
                raise InferenceError()
                if type == 'int':
                    logger.error('integer constant choices are not supported')
                    raise InferenceError()
                expr = int(tokens.pop(0)), int(tokens.pop(0)), int(tokens.pop(0)), int(tokens.pop(0))
                constants.add(expr)
            else:
                raise InferenceError()

    outputs = dict()
    for name, type in output_type.items():
        for i in range(0, len(output_instances[name])):
            if i not in output_instances[name]:
                logger.warn('output instance {} for variable {} is missing'.format(i, name))
                raise InferenceError()
        outputs[name] = (type, len(output_instances[name]))

    choices = dict()
    for expr, type in choice_type.items():
        for i in range(0, len(choice_instances[expr])):
            if i not in choice_instances[expr]:
                logger.warn('choice instance {} for variable {} is missing'.format(i, name))
                raise InferenceError()
        choices[expr] = (type, len(choice_instances[expr]), list(choice_env[expr]))

    return outputs, choices, constants, reachable, original


class SymbolicInferrer:

    def __init__(self, config, tester, load, searchDir):
        self.config = config
        self.run_test = tester
        self.load = load
        self.searchDir = searchDir
        self.trace_num = 0

    def _reduce_angelic_forest(self, angelic_paths):
        '''reduce the size of angelic forest (select shortest paths)'''
        logger.info('reducing angelic forest size from {} to {}'.format(len(angelic_paths),
                                                                        self.config['max_angelic_paths']))
        sorted_af = sorted(angelic_paths, key=len)
        return sorted_af[:self.config['max_angelic_paths']]

    def _boolean_angelic_forest(self, angelic_paths):
        '''convert all angelic values to booleans'''
        baf = []
        for path in angelic_paths:
            bpath = dict()
            for expr, instances in path.items():
                bpath[expr] = []
                for angelic, original, env_values in instances:
                    bpath[expr].append((bool(angelic), original, env_values))
            baf.append(bpath)
        return baf

    def get_oracle(self, dump, test):
        # name -> value list
        oracle = dict()
        vars = os.listdir(dump)
        logger.debug('dump: {}'.format(dump))
        logger.debug('vars: {}'.format(vars))
        for var in vars:
            instances = os.listdir(join(dump, var))
            for i in range(0, len(instances)):
                if str(i) not in instances:
                    logger.error('corrupted dump for test \'{}\''.format(test))
                    raise InferenceError()
            oracle[var] = []
            for i in range(0, len(instances)):
                file = join(dump, var, str(i))
                with open(file) as f:
                    content = f.read()
                oracle[var].append(content)
        return oracle

    def get_dump_parser_by_type(self):
        def str_to_int(s):
            return int(s)

        def str_to_long(s):
            return int(s)

        def str_to_bool(s):
            if s == 'false':
                return False
            if s == 'true':
                return True
            raise InferenceError()

        def str_to_char(s):
            if len(s) != 1:
                raise InferenceError()
            return s[0]
        dump_parser_by_type = dict()
        dump_parser_by_type['int'] = str_to_int
        dump_parser_by_type['long'] = str_to_long
        dump_parser_by_type['bool'] = str_to_bool
        dump_parser_by_type['char'] = str_to_char
        return dump_parser_by_type

    def get_to_bv_converter_by_type(self):
        def bool_to_bv(b):
            if b:
                return BitVecVal(1, 32)
            else:
                return BitVecVal(0, 32)

        def int_to_bv(i):
            return BitVecVal(i, 32)

        def long_to_bv(i):
            return BitVecVal(i, 64)

        def char_to_bv(c):
            return BitVecVal(ord(c), 32)
        to_bv_converter_by_type = dict()
        to_bv_converter_by_type['bool'] = bool_to_bv
        to_bv_converter_by_type['int'] = int_to_bv
        to_bv_converter_by_type['long'] = long_to_bv
        to_bv_converter_by_type['char'] = char_to_bv
        return to_bv_converter_by_type

    def get_from_bv_converter_by_type(self):
        def bv_to_bool(bv):
            return bv.as_long() != 0

        def bv_to_int(bv):
            v = bv.as_long()
            if v >> 31 == 1:  # negative
                v -= pow(2, 32)
            return v

        def bv_to_long(bv):
            v = bv.as_long()
            if v >> 63 == 1:  # negative
                v -= pow(2, 64)
            return v

        def bv_to_char(bv):
            v = bv.as_long()
            return chr(v)

        from_bv_converter_by_type = dict()
        from_bv_converter_by_type['bool'] = bv_to_bool
        from_bv_converter_by_type['int'] = bv_to_int
        from_bv_converter_by_type['long'] = bv_to_long
        from_bv_converter_by_type['char'] = bv_to_char
        return from_bv_converter_by_type

    def get_variables(self, path):
        variables = [str(var) for var in z3util.get_vars(path)
                     if str(var).startswith('int!')
                     or str(var).startswith('long!')
                     or str(var).startswith('bool!')
                     or str(var).startswith('char!')
                     or str(var).startswith('reachable!')]
        return variables

    def parse_smt2_file(self, smt):
        try:
            path = z3.parse_smt2_file(smt)
            return path
        except Exception:
            raise Smt2FileParseException()

    def parse_variables(self, vars):
        try:
            return parse_variables(vars)
        except Exception:
            raise Smt2VarParseException()

    def update_constraint(self, oracle_constraints, outputs,
                          oracle, reachable, dump_parser_by_type):
        logger.debug('outputs: {}'.format(outputs))
        for expected_variable, expected_values in oracle.items():
            if not self.config['ignore_unmatched_execution'] and expected_variable == 'reachable':
                expected_reachable = set(expected_values)
                if not (expected_reachable == reachable):
                    logger.info('labels \'{}\' executed while {} required'.format(
                        list(reachable),
                        list(expected_reachable)))
                    raise UnmatchedExecution()
                continue
            if expected_variable not in outputs.keys():
                outputs[expected_variable] = (None, 0)  # unconstraint does not mean wrong
            required_executions = len(expected_values)
            actual_executions = outputs[expected_variable][1]
            logger.debug('required_executions: {}'.format(required_executions))
            logger.debug('actual_executions: {}'.format(actual_executions))
            if not self.config['ignore_unmatched_execution'] and required_executions != actual_executions:
                logger.info('value \'{}\' executed {} times while {} required'.format(
                    expected_variable,
                    actual_executions,
                    required_executions))
                raise UnmatchedExecution()
            oracle_constraints[expected_variable] = []
            for i in range(0, required_executions):
                type = outputs[expected_variable][0]
                try:
                    value = dump_parser_by_type[type](expected_values[i])
                except Exception:
                    logger.error('variable \'{}\' has incompatible type {}'.format(expected_variable,
                                                                                   type))
                    raise InferenceError()
                oracle_constraints[expected_variable].append(value)
        return oracle_constraints, outputs

    def array_to_bv32(self, array):
        return Concat(Select(array, BitVecVal(3, 32)),
                      Select(array, BitVecVal(2, 32)),
                      Select(array, BitVecVal(1, 32)),
                      Select(array, BitVecVal(0, 32)))

    def array_to_bv64(self, array):
        return Concat(Select(array, BitVecVal(7, 32)),
                      Select(array, BitVecVal(6, 32)),
                      Select(array, BitVecVal(5, 32)),
                      Select(array, BitVecVal(4, 32)),
                      Select(array, BitVecVal(3, 32)),
                      Select(array, BitVecVal(2, 32)),
                      Select(array, BitVecVal(1, 32)),
                      Select(array, BitVecVal(0, 32)))

    def angelic_variable(self, type, expr, instance):
        pattern = '{}!choice!{}!{}!{}!{}!{}!angelic'
        s = pattern.format(type, expr[0], expr[1], expr[2], expr[3], instance)
        return Array(s, BitVecSort(32), BitVecSort(8))

    def original_variable(self, type, expr, instance):
        pattern = '{}!choice!{}!{}!{}!{}!{}!original'
        s = pattern.format(type, expr[0], expr[1], expr[2], expr[3], instance)
        return Array(s, BitVecSort(32), BitVecSort(8))

    def env_variable(self, expr, instance, name):
        pattern = 'int!choice!{}!{}!{}!{}!{}!env!{}'
        s = pattern.format(expr[0], expr[1], expr[2], expr[3], instance, name)
        return Array(s, BitVecSort(32), BitVecSort(8))

    def output_variable(self, type, name, instance):
        s = '{}!output!{}!{}'.format(type, name, instance)
        if type == 'long':
            return Array(s, BitVecSort(32), BitVecSort(8))
        else:
            return Array(s, BitVecSort(32), BitVecSort(8))

    def angelic_selector(self, expr, instance):
        s = 'angelic!{}!{}!{}!{}!{}'.format(expr[0], expr[1], expr[2], expr[3], instance)
        return BitVec(s, 32)

    def original_selector(self, expr, instance):
        s = 'original!{}!{}!{}!{}!{}'.format(expr[0], expr[1], expr[2], expr[3], instance)
        return BitVec(s, 32)

    def env_selector(self, expr, instance, name):
        s = 'env!{}!{}!{}!{}!{}!{}'.format(name, expr[0], expr[1], expr[2], expr[3], instance)
        return BitVec(s, 32)

    def update_solver(self, solver, path, oracle_constraints, outputs,
                      to_bv_converter_by_type, choices):
        solver.add(path)
        # logger.debug('add constraint:\n {}'.format(path))
        for name, values in oracle_constraints.items():
            type, _ = outputs[name]
            for i, value in enumerate(values):
                array = self.output_variable(type, name, i)
                bv_value = to_bv_converter_by_type[type](value)
                if type == 'long':
                    solver.add(bv_value == self.array_to_bv64(array))
                    # logger.debug('add constraint:\n {} == {}'.
                    #              format(bv_value, self.array_to_bv64(array)))
                else:
                    solver.add(bv_value == self.array_to_bv32(array))
                    # logger.debug('add constraint:\n {} == {}'.
                    #              format(bv_value, self.array_to_bv32(array)))

        for (expr, item) in choices.items():
            type, instances, env = item
            for instance in range(0, instances):
                selector = self.angelic_selector(expr, instance)
                array = self.angelic_variable(type, expr, instance)
                solver.add(selector == self.array_to_bv32(array))
                # logger.debug('add constraint:\n {} == {}'.
                #              format(selector, self.array_to_bv32(array)))

                selector = self.original_selector(expr, instance)
                array = self.original_variable(type, expr, instance)
                solver.add(selector == self.array_to_bv32(array))
                # logger.debug('add constraint:\n {} == {}'.
                #              format(selector, self.array_to_bv32(array)))

                for name in env:
                    selector = self.env_selector(expr, instance, name)
                    array = self.env_variable(expr, instance, name)
                    solver.add(selector == self.array_to_bv32(array))
                    # logger.debug('add constraint:\n {} == {}'.
                    #              format(selector, self.array_to_bv32(array)))
        return solver

    def get_angelic_path(self, test, choices, model, from_bv_converter_by_type,
                         original_available):
        # expr -> (angelic * original * env) list
        angelic_path = dict()

        if os.path.exists(self.load[test]):
            shutil.rmtree(self.load[test])
        os.mkdir(self.load[test])

        for (expr, item) in choices.items():
            angelic_path[expr] = []
            type, instances, env = item

            expr_str = '{}-{}-{}-{}'.format(expr[0], expr[1], expr[2], expr[3])
            expression_dir = join(self.load[test], expr_str)
            if not os.path.exists(expression_dir):
                os.mkdir(expression_dir)

            for instance in range(0, instances):
                bv_angelic = model[self.angelic_selector(expr, instance)]
                angelic = from_bv_converter_by_type[type](bv_angelic)
                bv_original = model[self.original_selector(expr, instance)]
                original = from_bv_converter_by_type[type](bv_original)
                if original_available:
                    logger.info('expression {}[{}]: angelic = {}, original = {}'.format(expr,
                                                                                        instance,
                                                                                        angelic,
                                                                                        original))
                else:
                    logger.info('expression {}[{}]: angelic = {}'.format(expr,
                                                                         instance,
                                                                         angelic))
                env_values = dict()
                for name in env:
                    bv_env = model[self.env_selector(expr, instance, name)]
                    value = from_bv_converter_by_type['int'](bv_env)
                    env_values[name] = value

                if original_available:
                    angelic_path[expr].append((angelic, original, env_values))
                else:
                    angelic_path[expr].append((angelic, None, env_values))

                # Dump angelic path to dump folder
                instance_file = join(expression_dir, str(instance))
                with open(instance_file, 'w') as file:
                    if isinstance(angelic, bool):
                        if angelic:
                            file.write('1')
                        else:
                            file.write('0')
                    else:
                        file.write(str(angelic))
        return angelic_path

    def get_klee_env(self, project):
        environment = dict(os.environ)
        if self.config['klee_max_forks'] is not None:
            environment['ANGELIX_KLEE_MAX_FORKS'] = str(self.config['klee_max_forks'])
        if self.config['klee_max_depth'] is not None:
            environment['ANGELIX_KLEE_MAX_DEPTH'] = str(self.config['klee_max_depth'])
        if self.config['klee_search'] is not None:
            environment['ANGELIX_KLEE_SEARCH'] = self.config['klee_search']
        if self.config['klee_timeout'] is not None:
            environment['ANGELIX_KLEE_MAX_TIME'] = str(self.config['klee_timeout'])
        if self.config['klee_solver_timeout'] is not None:
            environment['ANGELIX_KLEE_MAX_SOLVER_TIME'] = str(self.config['klee_solver_timeout'])
        if self.config['klee_debug']:
            environment['ANGELIX_KLEE_DEBUG'] = 'YES'
        if self.config['klee_ignore_errors']:
            environment['KLEE_DISABLE_MEMORY_ERROR'] = 'YES'
        if self.config['gobble_klee_message']:
            environment['GOBLE_KLEE_MESSAGE'] = 'YES'
        environment['ANGELIX_KLEE_WORKDIR'] = project.dir
        environment['ANGELIX_SYMBOLIC_RUNTIME'] = 'ON'
        return environment

    def run_klee(self, project, test, environment):
        environment['CC'] = 'angelix-compiler --test' if self.config['use_gcc'] \
            else 'angelix-compiler --klee'

        klee_start_time = time.time()
        self.run_test(project, test, run_mode=RunMode.KLEE, env=environment)
        klee_end_time = time.time()
        klee_elapsed = klee_end_time - klee_start_time
        return klee_elapsed

    # klee with a single path
    def run_klee_sp(self, project, test, environment):
        environment['CC'] = 'angelix-compiler --test' if self.config['use_gcc'] \
            else 'angelix-compiler --klee'
        self.run_test(project, test, run_mode=RunMode.KLEE_SP, env=environment)
        klee_log_file = join(project.dir, 'klee.log')
        try:
            with open(klee_log_file, errors='ignore') as f:
                contents = f.read()
                if 'KLEE: ERROR' in contents and 'KLEE: done: completed paths' not in contents:
                    return False
        except IOError:
            logger.warning('{} does not exist'.format(klee_log_file))
            return False
        return True

    def get_smt_files(self, project):
        klee_out_dir = join(project.dir, 'klee-out-0')
        logger.info('checking whether {} exists...'.format(klee_out_dir))
        klee_out_available.clear()
        event_handler = FileSystemEventHandler()

        def on_created(event):
            logger.log('{} is created'.format(klee_out_dir))
            klee_out_available.set()

        event_handler.on_created = on_created

        observer = Observer()
        observer.schedule(event_handler, klee_out_dir)
        if not os.path.exists(klee_out_dir):
            logger.info('wait until klee out dir gets ready for {}s...'.
                        format(self.config['klee_out_dir_timeout']))
            klee_out_available.wait(self.config['klee_out_dir_timeout'])

        smt_glob = join(klee_out_dir, '*.smt2')
        smt_files = sorted(glob(smt_glob))

        err_glob = join(klee_out_dir, '*.err')
        err_files = glob(err_glob)

        err_list = []
        for err in err_files:
            err_list.append(os.path.basename(err).split('.')[0])

        non_error_smt_files = []
        for smt in smt_files:
            smt_id = os.path.basename(smt).split('.')[0]
            if smt_id not in err_list:
                non_error_smt_files.append(smt)

        if not self.config['ignore_infer_errors']:
            smt_files = non_error_smt_files

        if len(smt_files) == 0 and len(err_list) == 0:
            logger.warning('No paths explored')
            raise NoSmtError()

        if len(smt_files) == 0:
            logger.warning('No non-error paths explored')
            raise NoSmtError()

        logger.debug('smt_files: {}'.format(smt_files))
        return smt_files

    def get_angelic_paths(self, smt_files, oracle, test, validation_project):
        angelic_paths = []

        z3.set_param("timeout", self.config['path_solving_timeout'])

        solver = Solver()

        for smt in smt_files:
            logger.info('solving path {}'.format(relpath(smt)))
            try:
                path = z3.parse_smt2_file(smt)
                smt2_vars = self.get_variables(path)
                # logger.debug('smt2_vars: {}'.format(smt2_vars))
                outputs, choices, constants, \
                    reachable, original_available = parse_variables(smt2_vars)
                logger.debug('outputs: {}'.format(outputs))
                logger.debug('choices: {}'.format(choices))
                logger.debug('constants: {}'.format(constants))
                logger.debug('reachable: {}'.format(reachable))
                logger.debug('original_available: {}'.format(original_available))
                # name -> value list (parsed)
                oracle_constraints = dict()

                dump_parser_by_type = self.get_dump_parser_by_type()
                to_bv_converter_by_type = self.get_to_bv_converter_by_type()
                from_bv_converter_by_type = self.get_from_bv_converter_by_type()

                oracle_constraints, outputs = self.update_constraint(
                    oracle_constraints, outputs, oracle, reachable, dump_parser_by_type)

                solver.reset()
                solver = self.update_solver(solver, path, oracle_constraints, outputs,
                                            to_bv_converter_by_type, choices)
                # logger.debug('solver: {}'.format(solver))

                result = solver.check()
                logger.debug('solver result: {}'.format(result))
                if result != z3.sat:
                    logger.info('UNSAT')  # TODO: can be timeout
                    continue
                model = solver.model()
                # logger.debug('model: {}'.format(model))

                angelic_path = self.get_angelic_path(test, choices, model,
                                                     from_bv_converter_by_type,
                                                     original_available)
                logger.debug('angelic_path: {}'.format(angelic_path))

                # Run Tester to validate the dumped values
                if not self.config['skip_validating_angelic_path'] and \
                   len(angelic_path) > 0:
                    validated = self.run_test(validation_project, test, load=self.load[test])
                else:
                    validated = True

                if validated:
                    angelic_paths.append(angelic_path)
                else:
                    logger.info('spurious angelic path')
            except Smt2FileParseException:
                logger.warning('failed to parse {}'.format(smt))
                continue
            except Smt2FileParseException:
                logger.warning('failed to parse {}'.format(smt2_vars))
                continue
            except UnmatchedExecution:
                continue
            except z3types.Z3Exception as e:
                if self.config['ignore_z3_exception']:
                    continue
                else:
                    raise e

        if self.config['synthesis_bool_only']:
            angelic_paths = self._boolean_angelic_forest(angelic_paths)

        if self.config['max_angelic_paths'] is not None and \
           len(angelic_paths) > self.config['max_angelic_paths']:
            angelic_paths = self._reduce_angelic_forest(angelic_paths)
        else:
            logger.info('found {} angelic paths for test \'{}\''.format(len(angelic_paths), test))

        return angelic_paths

    def __call__(self, project, test, locations, dump, validation_project):
        logger.info('inferring specification for test \'{}\' through KLEE'.format(test))

        self.test = test
        self.locations = locations

        inference_start_time = time.time()

        env = self.get_klee_env(project)
        env['CC'] = 'angelix-compiler --test' if self.config['use_gcc'] \
            else 'angelix-compiler --klee'

        klee_elapsed = self.run_klee(project, test, env)
        statistics.data['time']['klee'] += klee_elapsed
        statistics.save()

        smt_files = self.get_smt_files(project)

        # loading dump

        # name -> value list
        oracle = self.get_oracle(dump, test)
        logger.debug('oracle: {}'.format(oracle))

        # solving path constraints
        solving_start_time = time.time()

        angelic_paths = self.get_angelic_paths(smt_files, oracle, test,
                                               validation_project)

        solving_end_time = time.time()
        solving_elpased = solving_end_time - solving_start_time
        statistics.data['time']['solving'] += solving_elpased

        inference_end_time = time.time()
        inference_elapsed = inference_end_time - inference_start_time
        statistics.data['time']['inference'] += inference_elapsed

        iter_stat = dict()
        iter_stat['locations'] = locations
        iter_stat['test'] = test
        iter_stat['time'] = dict()
        iter_stat['time']['klee'] = klee_elapsed
        iter_stat['time']['solving'] = solving_elpased
        iter_stat['paths'] = dict()
        iter_stat['paths']['explored'] = len(smt_files)
        iter_stat['paths']['angelic_found'] = len(angelic_paths) > 0
        iter_stat['paths']['angelic'] = len(angelic_paths)
        statistics.data['iterations']['symbolic'].append(iter_stat)

        statistics.save()

        return angelic_paths
