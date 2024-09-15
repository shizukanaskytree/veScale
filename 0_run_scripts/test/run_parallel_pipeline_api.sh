#!/bin/bash

### /root/vescale_prj/veScale

### Ensure that the necessary modules and dependencies are correctly imported and accessible. If you encounter ModuleNotFoundError or similar issues, verify that the PYTHONPATH is set correctly to include the directories where the modules are located:
export PYTHONPATH=/root/vescale_prj/veScale:$PYTHONPATH
export PYTHONPATH=/root/vescale_prj/veScale/test:$PYTHONPATH

# /root/vescale_prj/veScale/test/parallel/pipeline/api/test_simple_api.py
torchrun --nproc_per_node=1 test/parallel/pipeline/api/test_simple_api.py

# test/parallel/pipeline/api/test_schedule_engine.py
torchrun --nproc_per_node=1 test/parallel/pipeline/api/test_schedule_engine.py

torchrun --nproc_per_node=1 test/parallel/pipeline/api/test_pipe_single_stage_ops.py

torchrun --nproc_per_node=1 test/parallel/pipeline/api/test_pipe_engine_api.py
