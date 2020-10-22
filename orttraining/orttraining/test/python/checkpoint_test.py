import subprocess

if __name__ == '__main__':
    # assert subprocess.call(['python', 'save_state.py', 'single_node_full_precision']) == 0
    # assert subprocess.call(['python', 'save_state.py', 'single_node_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'save_state.py', 'data_parallelism_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'save_state.py', 'data_parallelism_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'save_state.py', 'distributed_zero_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'save_state.py', 'distributed_zero_mixed_precision']) == 0

    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_full_precision_into_single_node_full_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_single_node_mixed_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_single_node_full_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_full_precision_into_single_node_mixed_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_single_node_full_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_single_node_full_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision']) == 0
    assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision']) == 0
    # assert subprocess.call(['python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision']) == 0

    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_full_precision_into_data_parallelism_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision']) == 0

    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_full_precision_into_distributed_zero_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision']) == 0
    # assert subprocess.call(['mpirun', '-n', '4', '-x', 'NCCL_DEBUG=INFO', 'python', 'orttraining_test_checkpoint.py', 'test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision']) == 0
