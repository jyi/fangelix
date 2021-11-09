import os
import stat
from contextlib import contextmanager
import shutil
import signal
import z3


class NoSmtError(Exception):
    pass


class InferenceError(Exception):
    pass


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    hs = '{}h '.format(int(h)) if int(h) > 0 else ''
    ms = '{}m '.format(int(m)) if int(m) > 0 or int(h) > 0 else ''
    ss = '{}s'.format(int(s))
    return '{}{}{}'.format(hs, ms, ss)


# Temporary solution to implement get_vars. In principle, it should be available in z3util.py
class AstRefKey:
    def __init__(self, n):
        self.n = n

    def __hash__(self):
        return self.n.hash()

    def __eq__(self, other):
        return self.n.eq(other.n)

    def __repr__(self):
        return str(self.n)


class z3util:
    def askey(n):
        assert isinstance(n, z3.AstRef)
        return AstRefKey(n)

    def get_vars(f):
        r = set()

        def collect(f):
            if z3.is_const(f):
                if f.decl().kind() == z3.Z3_OP_UNINTERPRETED and not z3util.askey(f) in r:
                    r.add(z3util.askey(f))
            else:
                for c in f.children():
                    collect(c)
        collect(f)
        return r
# end


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


class IdGenerator:

    def __init__(self):
        self.next = 0

    def next(self):
        self.next = self.next + 1
        return self.next - 1


class TimeoutException(Exception):
    pass


class SynthesisFailure(Exception):
    pass


class DefectClass:

    @staticmethod
    def is_rhs(type):
        return type == DefectClass.rhs_type()

    @staticmethod
    def is_loop_cond(type):
        return type == DefectClass.loop_type()

    @staticmethod
    def is_if_cond(type):
        return type == DefectClass.if_type()

    # This is for KLEE
    @staticmethod
    def is_branch(type):
        return type == DefectClass.branch_type()

    @staticmethod
    def is_pointer(type):
        return type == DefectClass.pointer_type()

    @staticmethod
    def rhs_type():
        return 'A'

    @staticmethod
    def loop_type():
        return 'L'

    @staticmethod
    def if_type():
        return 'I'

    @staticmethod
    def branch_type():
        return 'B'

    @staticmethod
    def pointer_type():
        return 'P'

    @staticmethod
    def guard_type():
        return 'G'    


class LocationInfo:
    """
    example of loc: ('A', (11, 7, 11, 11))
    """

    @staticmethod
    def defect_class(info):
        return info[0]

    @staticmethod
    def location(info):
        return info[1]

    @staticmethod
    def loc_id(info):
        return '-'.join(map(str, LocationInfo.location(info)))

    @staticmethod
    def line(info):
        loc = LocationInfo.location(info)
        return loc[0]


class TraceInfo:
    """
    example of loc: 
    ('A', (11, 7, 11, 11), 0)
    ('P', (11, 7, 11, 11), 0, 8)
    """

    @staticmethod
    def defect_class(info):
        return info[0]

    @staticmethod
    def location(info):
        return info[1]

    @staticmethod
    def val(info):
        return info[2]

    @staticmethod
    def max_val(info):
        return info[3]

    @staticmethod
    def loc_id(info):
        return '-'.join(map(str, TraceInfo.location(info)))

    @staticmethod
    def line(info):
        loc = TraceInfo.location(info)
        return loc[0]


class KleeLocationInfo:
    """
    example of loc:
    ('B', ('/angelix/tests/assignment-if/.angelix/frontend/test.c', ' 425'), '1')
    ('P', ('/angelix/tests/assignment-if/.angelix/frontend/test.c', ' 425'), '1', '8')
    """

    @staticmethod
    def defect_class(info):
        return info[0]

    @staticmethod
    def location(info):
        return info[1]

    @staticmethod
    def file(info):
        return info[1][0]

    @staticmethod
    def line(info):
        return info[1][1]

    @staticmethod
    def value(info):
        return info[2]

    @staticmethod
    def max_value(info):
        return info[3]


class KleeLocation:
    """
    example:
    ('/angelix-experiments/.angelix/frontend/mpn/powm.c', '96788')
    """

    @staticmethod
    def file(info):
        return info[0]

    @staticmethod
    def location(info):
        return info[1]


# Note that this is UNIX only
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def flatten(list):
    return sum(list, [])


def unique(list):
    """Select unique elements (order preserving)"""
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]


def rm_force(action, name, exc):
    os.chmod(name, stat.S_IREAD)
    shutil.rmtree(name)


def env_log(logger, key, env):
    if key in env:
        logger.debug('{}: {}'.format(key, env[key]))
