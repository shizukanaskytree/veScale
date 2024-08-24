### successfully run: The error message suggests using torchrun, which is the updated and recommended utility for distributed training. Here’s how to use it:
torchrun --nproc_per_node=4 test/parallel/pipeline/api/test_pipe_engine_api.py

### successfully run: And for a specific test
# torchrun --nproc_per_node=4 test/parallel/pipeline/api/test_pipe_engine_api.py -k test_runtime_engine
