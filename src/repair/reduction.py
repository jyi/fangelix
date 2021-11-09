import logging
import random
from math import sqrt
from utils import LocationInfo as LI
from utils import TraceInfo as TI

logger = logging.getLogger(__name__)


class Reducer:

    def __init__(self, config):
        self.config = config

    def __call__(self, test_suite, positive, negative, expressions):
        '''
        test_suite: list of tests to reduce
        positive, negative: (test * trace) list
        trace: expression list

        computes config['initial_tests'] tests that maximally cover given expressions
        '''
        number = self.config['initial_tests']
        number_failing = int(sqrt(number))

        # this code was originally written for multiple files:
        source_name = ''
        self.set_source_dirs([source_name])

        # test id -> source name -> set of locations
        data = {}

        passing_tests = []
        failing_tests = []

        # selecting best tests:
        relevant = set(map(LI.location, expressions))

        for test, trace in positive:
            if test in test_suite:
                data[test] = dict()
                data[test][source_name] = set(map(TI.location, trace)) & relevant
                passing_tests.append(test)

        for test, trace in negative:
            if test in test_suite:
                data[test] = dict()
                data[test][source_name] = set(map(TI.location, trace)) & relevant
                failing_tests.append(test)

        current_coverage = {}
        for source in self.source_dirs:
            current_coverage[source] = set()

        logger.info("all failing tests: {}".format(failing_tests))
        if self.config['all_tests']:
            selected_failing = failing_tests
        else:
            selected_failing = self.select_best_tests(failing_tests, number_failing,
                                                      data, current_coverage)
        number_selected_failing = len(selected_failing)

        if self.config['all_tests']:
            selected_passing = passing_tests
        else:
            selected_passing = self.select_best_tests(passing_tests,
                                                      number - number_selected_failing,
                                                      data,
                                                      current_coverage)
        number_selected_passing = len(selected_passing)

        total_selected = number_selected_passing + number_selected_failing

        logger.info("selected {} tests".format(total_selected))
        logger.info("selected passing tests: {}".format(selected_passing))
        logger.info("selected failing tests: {}".format(selected_failing))

        if number_selected_failing == 0:
            logger.warning('no failing tests')
            exit(1)

        return selected_failing + selected_passing

    def select_best_tests(self, candidates, max_number, data, current_coverage):
        logger.info('select {} best tests from {}'.format(max_number, candidates))
        selected = []
        for i in range(0, max_number):
            if len(candidates) == 0:
                break
            best_increment = {}
            for source in self.source_dirs:
                best_increment[source] = 0

            best_increment_total = 0

            best_test = candidates[0]

            best_coverage = 0
            best_coverage_test = best_test

            for test in candidates:
                current_increment = {}
                current_increment_total = 0
                coverage = 0
                for source in self.source_dirs:
                    current_increment[source] = len(data[test][source] - current_coverage[source])
                    current_increment_total += current_increment[source]
                    coverage = coverage + len(data[test][source])
                if current_increment_total > best_increment_total:
                    best_increment = current_increment
                    best_increment_total = current_increment_total
                    best_test = test
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_coverage_test = test

            if best_increment_total > 0:
                selected.append(best_test)
                candidates.remove(best_test)
                for source in self.source_dirs:
                    current_coverage[source] |= data[best_test][source]
            elif best_coverage > 0:
                selected.append(best_coverage_test)
                candidates.remove(best_coverage_test)
                for source in self.source_dirs:
                    current_coverage[source] = data[best_coverage_test][source]
            else:
                break

        if len(selected) < max_number and len(candidates) > 0:
            delta = max_number - len(selected)
            logger.info('add {} tests at random'.format(delta))
            for i in range(delta):
                choice = random.choice(candidates)
                selected.append(choice)
                candidates.remove(choice)
        return selected

    def set_source_dirs(self, source_dirs):
        self.source_dirs = source_dirs
