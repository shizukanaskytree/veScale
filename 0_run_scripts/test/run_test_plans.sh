#!/bin/bash

### /root/vescale_prj/veScale

### Ensure that the necessary modules and dependencies are correctly imported and accessible. If you encounter ModuleNotFoundError or similar issues, verify that the PYTHONPATH is set correctly to include the directories where the modules are located:
export PYTHONPATH=/root/vescale_prj/veScale:$PYTHONPATH
export PYTHONPATH=/root/vescale_prj/veScale/test:$PYTHONPATH

# torchrun --nproc_per_node=4 test/dmodule/test_plans.py

torchrun --nproc_per_node=1 test/dmodule/test_plans_copy.py -k test_cuda
