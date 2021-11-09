from math import sqrt, ceil
from runtime import TraceItem
from utils import LocationInfo as LI
import logging


logger = logging.getLogger(__name__)


class NoNegativeTestException(Exception):
    pass


def ochiai(executed_passing, executed_failing, total_passing, total_failing):
    if not total_failing > 0:
        raise NoNegativeTestException()
    if executed_failing + executed_passing == 0:
        return 0
    return executed_failing / sqrt(total_failing * (executed_passing + executed_failing))


def jaccard(executed_passing, executed_failing, total_passing, total_failing):
    if not total_failing > 0:
        raise NoNegativeTestException()
    return executed_failing / (total_failing + executed_passing)


def tarantula(executed_passing, executed_failing, total_passing, total_failing):
    if not total_failing > 0:
        raise NoNegativeTestException()
    if executed_failing + executed_passing == 0:
        return 0
    return ((executed_failing / total_failing) /
            ((executed_failing / total_failing) + (executed_passing / total_passing)))


class Localizer:

    def __init__(self, config, lines):
        self.lines = lines
        self.config = config

    def __call__(self, test_suite, all_positive, all_negative):
        '''
        test_suite: tests under consideration
        all_positive, all_negative: (test * trace) list
        trace: expression list

        computes config['suspicious']/config['group_size'] groups
        each consisting of config['group_size'] suspicious expressions
        '''

        if self.config['localization'] == 'ochiai':
            formula = ochiai
        elif self.config['localization'] == 'jaccard':
            formula = jaccard
        elif self.config['localization'] == 'tarantula':
            formula = tarantula

        # first, remove irrelevant information:
        positive = []
        negative = []
        dc_idx = 0
        loc_idx = 1

        def del_val(pair):
            def _del_val(t):
                dc, loc, val = t
                return dc, loc
            test, trace = pair
            trace = list(map(_del_val, trace))
            return test, trace
        all_positive = list(map(del_val, all_positive))
        all_negative = list(map(del_val, all_negative))

        if not self.config['invalid_localization']:
            for test, trace in all_positive:
                if test in test_suite:
                    positive.append((test, trace))

            for test, trace in all_negative:
                if test in test_suite:
                    locs = [tuple([t[dc_idx], t[loc_idx]]) for t in trace]
                    negative.append((test, locs))
        else:
            positive = all_positive
            negative = all_negative

        # logger.debug('positive: {}'.format(positive))
        # logger.debug('negative: {}'.format(negative))

        all = set()

        for _, trace in positive:
            all |= set(trace)

        for _, trace in negative:
            all |= set(trace)

        executed_positive = dict()
        executed_negative = dict()

        for e in all:
            executed_positive[e] = 0
            executed_negative[e] = 0

        for _, trace in positive:
            executed = set(trace)
            for e in executed:
                executed_positive[e] += 1

        for _, trace in negative:
            executed = set(trace)
            for e in executed:
                executed_negative[e] += 1

        with_score = []

        logger.debug('all: {}'.format(all))
        logger.debug('lines: {}'.format(self.lines))
        if self.lines is not None:
            filtered = filter(lambda item: TraceItem.get_location(item)[0] in self.lines, all)
            all = list(filtered)

        logger.debug('filtered all: {}'.format(all))
        logger.debug('executed_positive: {}'.format(executed_positive))
        logger.debug('executed_negative: {}'.format(executed_negative))

        logger.debug('total_passing: {}'.format(len(positive)))
        logger.debug('total_failing: {}'.format(len(negative)))
        for e in all:
            try:
                if e in executed_negative:
                    score = formula(executed_positive[e], executed_negative[e],
                                    len(positive), len(negative))
                    logger.debug('(loc, score) = ({}, {})'.format(e, score))
                    with_score.append((e, score))
            except NoNegativeTestException:
                logger.info("No negative test exists")
                exit(0)

        ranking = sorted(with_score, key=lambda r: r[1], reverse=True)
        logger.debug('ranking: {}'.format(ranking))

        if self.config['additional_susp_locs'] is not None:
            logger.debug('add additional suspicious locations')
            default_score = self.config['default_susp_score']
            for info in self.config['additional_susp_locs']:
                # e.g., info: A-293-7-293-7
                dc, loc = info.split('-', 1)
                loc_tuple = tuple(map(int, loc.split('-')))
                ranking.append(((dc, loc_tuple), default_score))

        if len(ranking) == 0:
            logger.warning('no location is assigned a score')
            logger.debug('executed_positive: {}'.format(executed_positive))
            logger.debug('executed_negative: {}'.format(executed_negative))

        if self.config['show_suspicious_locations']:
            for (loc, score) in ranking:
                logger.info('(loc, score) = ({}, {})'.format(loc, score))

        logger.debug('all_suspicious: {}'.format(self.config['all_suspicious']))
        if self.config['all_suspicious']:
            suspicious = len(ranking)
        else:
            suspicious = self.config['suspicious']
        logger.debug('suspicious: {}'.format(suspicious))

        if self.config['group_by_score']:
            top = ranking[:suspicious]
        else:
            if self.config['localize_from_bottom']:
                # sort by location backward
                top = sorted(ranking[:suspicious], key=lambda r: LI.line(r[0]), reverse=True)
            else:
                # sort by location
                top = sorted(ranking[:suspicious], key=lambda r: LI.line(r[0]))

        logger.debug('top: {}'.format(top))
        group_size = self.config['group_size'] if not self.config['single_group'] \
                     else suspicious
        groups_with_score = []
        for i in range(0, ceil(suspicious / group_size)):
            if len(top) == 0:
                break
            group = []
            total_score = 0
            for j in range(0, group_size):
                if len(top) == 0:
                    break
                expr, score = top.pop(0)
                total_score += score
                group.append(expr)
            groups_with_score.append((group, total_score))

        sorted_groups = sorted(groups_with_score, key=lambda r: r[1], reverse=True)

        if self.config['show_suspicious_locations']:
            for idx, (group, score) in enumerate(sorted_groups):
                logger.info('group {}: {} ({})'.format(idx + 1, group, score))

        groups = []
        for (group, score) in sorted_groups:
            groups.append(group)
            logger.info("selected expressions {} with group score {:.5} ".format(group, score))

        logger.debug('groups: {}'.format(groups))
        return groups, dict(ranking)
