This folder comprises the code and simulation files featured in the paper: On Heterogeneous Treatment Effects in Heterogeneous Causal Graphs

In order to run ISL from the command line use cl.py. For example to run one replicate of scenario 1 for 10 samples use:

python cl.py --scenario S1 --sample_size 10

to discover a graph for specific data use:

python cl.py --infile data.csv --p NUMBER_OF_POTENTIAL_MODERATORS --s NUMBER_OF_POTENTIAL_MEDIATORS

to run two replicates of scenario 1 for 10 samples use:

python cl.py --scenario S1 --sample_size 10 --rep_number 2

to start from a specific seed use:

python cl.py --scenario S1 --sample_size 10 --rep_number 2 --start 42

to end at a specific seed use:

python cl.py --scenario S1 --sample_size 10 --start 42 --end 142

or equivalently

python cl.py --scenario S1 --sample_size 10 --start 42 --rep_number 101

finally, to use multiple cores use:

python cl.py --scenario S1 --sample_size 10 --start 42 --rep_number 101 --num_cores 4

Use python cl.py --help for more commands.

Note that the real AURORA dataset cannot be shared due to the AURORA privacy protocol.
