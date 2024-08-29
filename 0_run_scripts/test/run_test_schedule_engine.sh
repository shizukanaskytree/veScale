#!/bin/bash

### /root/vescale_prj/veScale

### Ensure that the necessary modules and dependencies are correctly imported and accessible. If you encounter ModuleNotFoundError or similar issues, verify that the PYTHONPATH is set correctly to include the directories where the modules are located:
export PYTHONPATH=/root/vescale_prj/veScale:$PYTHONPATH
export PYTHONPATH=/root/vescale_prj/veScale/test:$PYTHONPATH

### successful run
torchrun --nproc_per_node=4 test/parallel/pipeline/api/test_schedule_engine.py

### successful run
torchrun --nproc_per_node=4 test/parallel/pipeline/api/test_schedule_engine.py -k test_simple_1f1b
