#!/bin/bash

### /root/vescale_prj/veScale

### Ensure that the necessary modules and dependencies are correctly imported and accessible. If you encounter ModuleNotFoundError or similar issues, verify that the PYTHONPATH is set correctly to include the directories where the modules are located:
export PYTHONPATH=/root/vescale_prj/veScale:$PYTHONPATH
export PYTHONPATH=/root/vescale_prj/veScale/test:$PYTHONPATH

### successful run
torchrun --nproc_per_node=4 test/parallel/pipeline/instruction/test_schedule.py

# The ndtimeline is used in the test_runtime_engine_with_profiling and test_zerobubble_engine methods of the PipelineScheduleTest class.
# In both methods, it is used to initialize and manage distributed nD timeline profiling for pipeline parallelism. Specifically, init_ndtimers, flush, and wait functions from vescale.ndtimeline are utilized:
# 	1.	init_ndtimers: Initializes the nD timers for profiling. It is configured with parameters such as rank, local_rank, and enable_streamer.
# 	2.	flush: Flushes the collected profiling data.
# 	3.	wait: Ensures synchronization and waits for all profiling operations to complete.
# These are typically used to profile and debug the performance of distributed training across different devices and stages in a parallelized deep learning model.

### zero bubble, ndtimeline
torchrun --nproc_per_node=1 test/parallel/pipeline/instruction/test_schedule.py -k test_zerobubble_engine

### ndtimeline
torchrun --nproc_per_node=4 test/parallel/pipeline/instruction/test_schedule.py -k test_runtime_engine_with_profiling

