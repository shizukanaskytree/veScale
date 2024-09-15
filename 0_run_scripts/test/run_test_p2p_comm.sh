#!/bin/bash

### /root/vescale_prj/veScale

# 这个test_decomposable_5d_parallelization例子好像是把 Pipeline Parallelism, parallelize_module, sharding_plan 都结合了

### Ensure that the necessary modules and dependencies are correctly imported and accessible. If you encounter ModuleNotFoundError or similar issues, verify that the PYTHONPATH is set correctly to include the directories where the modules are located:
export PYTHONPATH=/root/vescale_prj/veScale:$PYTHONPATH
export PYTHONPATH=/root/vescale_prj/veScale/test:$PYTHONPATH

### root@d3cb3edeb6a9:~/vescale_prj/veScale#
torchrun --nproc_per_node=1 test/parallel/pipeline/backend/test_p2p_comm.py # OK

torchrun --nproc_per_node=4 test/parallel/pipeline/backend/test_p2p_comm.py # OK

