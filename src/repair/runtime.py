import os
from os.path import join, exists
import shutil
from utils import rm_force
import logging
from utils import DefectClass as DC
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class Dump:

    def _json_to_dump(self, json):
        for test, data in json.items():
            test_dir = join(self.dir, test)
            if exists(test_dir):
                shutil.rmtree(test_dir, onerror=rm_force)
            os.mkdir(test_dir)
            for variable, values in data.items():
                variable_dir = join(test_dir, variable)
                if exists(variable_dir):
                    shutil.rmtree(variable_dir, onerror=rm_force)
                os.mkdir(variable_dir)
                for i, v in enumerate(values):
                    instance_file = join(variable_dir, str(i))
                    logger.debug('instance_file: {}'.format(instance_file))
                    with open(instance_file, 'w') as file:
                        file.write(str(v))

    def export(self):
        json = dict()
        tests = os.listdir(self.dir)
        for test in tests:
            dump = self[test]
            json[test] = dict()
            vars = os.listdir(dump)
            for var in vars:
                instances = os.listdir(join(dump, var))
                json[test][var] = []
                for i in range(0, len(instances)):
                    file = join(dump, var, str(i))
                    with open(file) as f:
                        content = f.read()
                    json[test][var].append(content)
                if var == 'reachable':
                    json[test][var] = list(set(json[test][var]))

        return json

    def __init__(self, working_dir, correct_output):
        self.dir = join(working_dir, 'dump')
        if exists(self.dir):
            shutil.rmtree(self.dir, onerror=rm_force)
        os.mkdir(self.dir)
        if correct_output is not None:
            self._json_to_dump(correct_output)

    def __iadd__(self, test_id):
        dir = join(self.dir, test_id)
        if exists(dir):
            shutil.rmtree(dir, onerror=rm_force)
        os.mkdir(dir)
        return self

    def add_test(self, test_id):
        dir = join(self.dir, test_id)
        if not exists(dir):
            os.mkdir(dir)
        return self

    def __getitem__(self, test_id):
        dir = join(self.dir, test_id)
        return dir

    def __contains__(self, test_id):
        dir = join(self.dir, test_id)
        if exists(dir):
            return True
        else:
            return False


class Trace:

    def __init__(self, working_dir):
        self.dir = join(working_dir, 'trace')
        if exists(self.dir):
            shutil.rmtree(self.dir, onerror=rm_force)
        os.mkdir(self.dir)

    def __iadd__(self, test_id):
        trace_file = join(self.dir, test_id)
        file = open(trace_file, 'w')
        file.close()
        return self

    def __getitem__(self, test_id):
        trace_file = join(self.dir, test_id)
        return trace_file

    def __contains__(self, test_id):
        trace_file = join(self.dir, test_id)
        if exists(trace_file):
            return True
        else:
            return False

    @staticmethod
    def parse_trace_file(trace_file):
        logger.debug('parse {}'.format(trace_file))
        trace = []
        dc_idx = 0
        loc_idx = 1
        val_idx = 2
        max_idx = 4
        if os.path.exists(trace_file):
            with open(trace_file) as file:
                for line in file:
                    ss = line.split(',')
                    dc = ss[dc_idx].strip()
                    loc = ss[loc_idx].strip()
                    val = ss[val_idx].strip()
                    id = [int(s) for s in loc.split('-')]
                    assert len(id) == 4
                    if dc == DC.pointer_type():
                        max_val = ss[max_idx].strip()
                        trace.append((dc, tuple(id), val, max_val))
                    else:
                        trace.append((dc, tuple(id), val))
        else:
            logger.info('No trace file: {}'.format(trace_file))
        return trace

    @staticmethod
    def parse_klee_trace_file(trace_file):
        logger.debug('parse {}'.format(trace_file))
        trace = []
        type_idx = 0
        file_idx = 1
        loc_idx = 2
        val_idx = 4
        max_idx = 5

        if os.path.exists(trace_file):
            with open(trace_file) as file:
                for line in file:
                    ss = line.split(',')
                    dc = ss[type_idx].strip()
                    file_name = ss[file_idx].strip()
                    loc = ss[loc_idx].strip()
                    val = ss[val_idx].strip()
                    if DC.is_pointer(dc):
                        max_val = ss[max_idx].strip()
                        trace.append((dc, (file_name, loc), val, max_val))
                    else:
                        trace.append((dc, (file_name, loc), val))
        else:
            logger.info('No trace file: {}'.format(trace_file))

        return trace

    '''
    loc: e.g., 10-10-10-14
    return (10, 10, 10, 14)
    '''
    @staticmethod
    def parseLoc(loc):
        l1, l2, l3, l4 = loc.split('-', maxsplit=4)
        return (int(l1), int(l2), int(l3), int(l4))

    '''
    ctxt: e.g., n = 2 ; x = 1
    return: {'n': 2, 'x': 1}
    '''
    @staticmethod
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

    def parse(self, test_id):
        trace_file = join(self.dir, test_id)
        return Trace.parse_trace_file(trace_file)


class TraceItem:

    defect_class_idx = 0
    loc_idx = 1
    val_idx = 2

    @staticmethod
    def get_defect_class(item):
        return item[TraceItem.defect_class_idx]

    @staticmethod
    def get_location(item):
        return item[TraceItem.loc_idx]

    @staticmethod
    def get_value(item):
        return item[TraceItem.val_idx]

    @staticmethod
    def is_cond(item):
        dc = TraceItem.get_defect_class(item)
        return dc == DC.if_type() or dc == DC.loop_type() or dc == DC.guard_type()

    @staticmethod
    def is_ptr(item):
        dc = TraceItem.get_defect_class(item)
        return dc == DC.pointer_type()


class Load:

    def __init__(self, working_dir):
        self.dir = join(working_dir, 'load')
        if exists(self.dir):
            shutil.rmtree(self.dir, onerror=rm_force)
        os.mkdir(self.dir)

    def __getitem__(self, test_id):
        trace_file = join(self.dir, test_id)
        return trace_file


class SearchDir:

    def __init__(self, working_dir):
        self.dir = join(working_dir, 'search')
        if exists(self.dir):
            shutil.rmtree(self.dir, onerror=rm_force)
        os.mkdir(self.dir)

    def __getitem__(self, test_id):
        trace_file = join(self.dir, 'test_' + test_id)
        return trace_file


class SynDir:

    def __init__(self, working_dir):
        self.dir = join(working_dir, 'syn')
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

    def __getitem__(self, id):
        syn_dir = join(self.dir, 'syn_' + str(id))
        if not os.path.exists(syn_dir):
            os.mkdir(syn_dir)
        return syn_dir


class DeltaDebuggingDir:

    def __init__(self, working_dir):
        self.dir = join(working_dir, 'DD')
        if exists(self.dir):
            shutil.rmtree(self.dir, onerror=rm_force)
        os.mkdir(self.dir)

    def __getitem__(self, test_id):
        trace_file = join(self.dir, 'test_' + test_id)
        return trace_file
