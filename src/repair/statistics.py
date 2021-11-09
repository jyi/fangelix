import time
import json
from os.path import join

data = dict()


def init(working_dir, config):
    data['file'] = join(working_dir, 'statistics.json')
    data['src'] = 'UNKNOWN'
    data['buggy'] = 'UNKNOWN'
    data['patch_found'] = 'UNKNOWN'
    data['patch_file'] = 'UNKNOWN'
    data['config'] = config
    data['time'] = dict()
    data['time']['testing'] = 0
    data['time']['compilation'] = 0
    data['time']['klee'] = 0
    data['time']['synthesis'] = 0
    data['time']['solving'] = 0
    data['time']['inference'] = 0
    data['time']['dd'] = 0
    data['time']['total'] = -1
    data['iterations'] = dict()
    data['iterations']['synthesis'] = []
    data['iterations']['guided'] = []
    data['iterations']['symbolic'] = []
    data['iterations']['random'] = []


def save():
    with open(data['file'], 'w') as output_file:
        asserts = json.dump(data, output_file, indent=2)
