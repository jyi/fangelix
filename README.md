# FAngelix #

FAngelix is a program repair tool for C programs. It is an extension of [Angelix](https://github.com/mechtaev/angelix), a constraint-based (semantics-based) repair tool. FAngelix uses a guided search algorithm based on [MCMC \(Markov Chain Monte Carlo\) sampling](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) and is generally faster than Angelix.

## Installation ##

The installation steps can be identified with the provided [Dockerfile](https://github.com/jyi/fangelix/edit/master/Dockerfile).

## Running FAngelix ##

Before running FAngelix, make sure to run the following to set FAngelix environment:

    . activate
    
FAngelix is implemented as an extension of Angelix, keeping the same command-line interface of Angelix. Let's fix a buggy source code, [test.c](https://github.com/jyi/angelix/blob/fangelix-release/tests/loop-condition/src/test.c), available in the test/loop-condition directory with FAngelix as follows. 

    # Assume that we are in the root directory of fangelix.
    cd tests/loop-condition
    angelix src test.c oracle 1 2 3


The first parameter, `src`, describes the directory where the buggy source code `test.c` is located. The [`oracle`](https://github.com/jyi/angelix/blob/fangelix-release/tests/loop-condition/oracle) and `1 2 3` describe a test script and test IDs. FAngelix performs a guided search based on the cost of executions and the [`oracle`](https://github.com/jyi/angelix/blob/fangelix-release/tests/loop-condition/oracle) contains a cost function named `cal_diff`. If a cost function is not provided, FAngelix performs without a cost function. For the above example, FAngelix will finish with output similar to the following:


    INFO     synthesis       fixing expression (30, 10, 30, 14): (n > 1) ---> (n >= 1)
    INFO     repair          candidate fix synthesized
    INFO     transformation  applying patch to validation source
    INFO     project         building validation source
    INFO     testing         running test '1' of validation source
    INFO     testing         output code: 0
    INFO     testing         running test '2' of validation source
    INFO     testing         output code: 0
    INFO     testing         running test '3' of validation source
    INFO     testing         output code: 0
    INFO     repair          patch successfully generated in 14s (see src-2021-Nov08-175730.patch)
    SUCCESS

FAngelix by default fixes two defect classes: if conditions and loop conditions. To fix an assignment, use `--defect assignments` as follows:

    cd tests/assignment-if
    angelix src test.c oracle 1 2 3 --assert assert.json --defect assignments

When fixing an assignment, FAngelix uses symbolic execution tool KLEE and the expected output needs to be specified separately in the `assert.json` file. For a detailed description, please refer to the [tutorial of Angelix](https://github.com/jyi/angelix/blob/fangelix-release/doc/Tutorial.md). For the second [example](https://github.com/jyi/angelix/blob/fangelix-release/tests/assignment-if/src/test.c), FAngelix will finish with output similar to the following:

    INFO     synthesis       fixing expression (13, 7, 13, 11): (a + b) ---> (a - b)
    INFO     repair          candidate fix synthesized
    INFO     transformation  applying patch to validation source
    INFO     project         building validation source
    INFO     testing         running test '1' of validation source
    INFO     testing         output code: 0
    INFO     testing         running test '2' of validation source
    INFO     testing         output code: 0
    INFO     testing         running test '3' of validation source
    INFO     testing         output code: 0
    INFO     repair          patch successfully generated in 36s (see src-2021-Nov08-180056.patch)
    SUCCESS


FAngelix maintains the original algorithm of Angelix. The following command can be used to use the Angelix mode instead.

    cd tests/loop-condition
    angelix src test.c oracle 1 2 3 --assert assert.json --angelic-search-strategy symbolic --klee-timeout 10

## Experimental scripts and data ##

Experimental scripts and data are available [here](https://github.com/jyi/angelix-experiments).
