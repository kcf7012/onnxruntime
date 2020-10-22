import os
import pickle
from numpy.testing import assert_allclose
import numpy as np
import argparse
import glob

import torch
import torch.distributed as dist

import onnxruntime
from onnxruntime import set_seed
from onnxruntime.training import amp, checkpoint, optim, orttrainer
from onnxruntime.capi._pybind_state import set_cuda_device_id, get_mpi_context_world_rank, get_mpi_context_world_size
from orttraining_test_orttrainer_frontend import _load_pytorch_transformer_model

def distributed_setup(save_function):
    def setup():
        world_rank = get_mpi_context_world_rank()
        world_size = get_mpi_context_world_size()
        device = 'cuda:' + str(world_rank)

        os.environ['RANK'] = str(world_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        set_cuda_device_id(world_rank)

        dist.init_process_group(backend='nccl', world_size=world_size, rank=world_rank)
        save_function(world_rank, world_size, device)
    return setup

def test_load_from_single_node_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_single_node_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert_allclose(state_dict_pre_checkpoint[full_precision_key], state_dict_pre_checkpoint[key], atol=1e-3)
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_single_node_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_single_node_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_distributed_zero_full_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        print('==============================')
        print('Rank: ', rank)
        print('precheckpoint: ', state_dict_pre_checkpoint.keys())
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        print('postcheckpoint: ', state_dict_post_checkpoint.keys())
        print('==============================')
        for key, value in state_dict_pre_checkpoint.items():
            if key not in state_dict_post_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_post_checkpoint[key])
        for key, value in state_dict_post_checkpoint.items():
            if key not in state_dict_pre_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_pre_checkpoint[key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_post_checkpoint[key])

# not working
def test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {'device' : {'id' : device},
            'debug' : {'deterministic_compute': True}}
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        for key, value in state_dict_pre_checkpoint.items():
            if key.endswith('_fp16'):
                full_precision_key = key[:-5]
                assert full_precision_key in state_dict_post_checkpoint
                # if not np.allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3):
                #     print(key)
                #     continue
                assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
                continue
            elif key not in state_dict_post_checkpoint:
                assert key.startswith('Moment')
                continue
            elif state_dict_pre_checkpoint[key].dtype == torch.float32:
                continue
            assert_allclose(value, state_dict_post_checkpoint[key])
        # for key, value in state_dict_post_checkpoint.items():
        #     if key not in state_dict_pre_checkpoint:
        #         assert key.startswith('Moment')
        #         continue
        #     assert_allclose(value, state_dict_pre_checkpoint[key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_post_checkpoint[key])

def test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to initialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        for key, value in state_dict_pre_checkpoint.items():
            if key not in state_dict_post_checkpoint:
                assert key.startswith('Moment')
                continue
            elif state_dict_pre_checkpoint[key].dtype == torch.float32:
                continue
            if key.endswith('_fp16'):
                full_precision_key = key[:-5]
                assert_allclose(state_dict_post_checkpoint[key], state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            assert_allclose(value, state_dict_post_checkpoint[key])
        for key, value in state_dict_post_checkpoint.items():
            if state_dict_post_checkpoint[key].dtype == torch.float32:
                continue
            elif key not in state_dict_pre_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_pre_checkpoint[key])

def test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision(device = 'cuda', checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        for key, value in state_dict_post_checkpoint.items():
            if key.endswith('_fp16'):
                full_precision_key = key[:-5]
                assert full_precision_key in state_dict_pre_checkpoint
                assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
                continue
            elif key not in state_dict_pre_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to initialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
        assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to initialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])
    
    # load state into pytorch and compare
    model.load_state_dict(state_dict_pre_checkpoint, strict=False)
    state_dict_pytorch = model.state_dict()
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        for key, value in state_dict_pre_checkpoint.items():
            if key not in state_dict_post_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_post_checkpoint[key])
        for key, value in state_dict_post_checkpoint.items():
            if key not in state_dict_pre_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_pre_checkpoint[key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_post_checkpoint[key])

# not working
@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        for key, value in state_dict_pre_checkpoint.items():
            if key.endswith('_fp16'):
                full_precision_key = key[:-5]
                assert full_precision_key in state_dict_post_checkpoint
                if not np.allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3):
                    print(key)
                    continue
                assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
                continue
            elif key not in state_dict_post_checkpoint:
                assert key.startswith('Moment')
                continue
            elif state_dict_pre_checkpoint[key].dtype == torch.float32:
                continue
            assert_allclose(value, state_dict_post_checkpoint[key])
    
    # load state into pytorch and compare
    checkpoint_files = checkpoint._list_checkpoint_files(checkpoint_dir, "ORT_checkpoint")
    agg_checkpoint = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    agg_state_dict = agg_checkpoint.aggregate_checkpoints()
    model.load_state_dict(agg_state_dict, strict=False)
    state_dict_pytorch = model.state_dict()
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pytorch.items():
        assert state_dict_pytorch[key].dtype == torch.float32
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to initialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        for key, value in state_dict_pre_checkpoint.items():
            if state_dict_pre_checkpoint[key].dtype == torch.float32:
                continue
            elif key not in state_dict_post_checkpoint:
                assert key.startswith('Moment')
                continue            
            assert_allclose(value, state_dict_post_checkpoint[key])
        for key, value in state_dict_post_checkpoint.items():
            if state_dict_post_checkpoint[key].dtype == torch.float32:
                continue
            elif key not in state_dict_pre_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state_counter = len(glob.glob1(checkpoint_dir,"state_dict*"))
    for rank in range(state_counter):
        state = pickle.load(open(checkpoint_dir+'state_dict_'+str(rank)+'.pkl', 'rb'))

        # compare all states
        state_dict_pre_checkpoint = state['state_dict_'+str(rank)]
        state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
        for key, value in state_dict_post_checkpoint.items():
            if key.endswith('_fp16'):
                full_precision_key = key[:-5]
                assert full_precision_key in state_dict_pre_checkpoint
                assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
                continue
            elif key not in state_dict_pre_checkpoint:
                assert key.startswith('Moment')
                continue
            assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key not in state_dict_post_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    for key, value in state_dict_post_checkpoint.items():
        if key not in state_dict_pre_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            continue
        elif key not in state_dict_post_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if state_dict_pre_checkpoint[key].dtype == torch.float32:
            continue
        elif key not in state_dict_post_checkpoint:
            print(key)
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    for key, value in state_dict_post_checkpoint.items():
        if state_dict_post_checkpoint[key].dtype == torch.float32:
            continue
        elif key not in state_dict_pre_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/single_node/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to initialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
            continue
        elif key not in state_dict_pre_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key not in state_dict_post_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    for key, value in state_dict_post_checkpoint.items():
        if key not in state_dict_pre_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            continue
        elif key not in state_dict_post_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if state_dict_pre_checkpoint[key].dtype == torch.float32:
            continue
        elif key not in state_dict_post_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    for key, value in state_dict_post_checkpoint.items():
        if state_dict_post_checkpoint[key].dtype == torch.float32:
            continue
        elif key not in state_dict_pre_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/data_parallelism/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to initialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict']
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
            continue
        elif key not in state_dict_pre_checkpoint:
            assert key.startswith('Moment')
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        assert_allclose(value, state_dict_post_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    print('==========================================================')
    for key, value in state_dict_pre_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_post_checkpoint
            if not np.allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3):
                print(key)
                continue
            assert_allclose(value, state_dict_post_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    print('==========================================================')
    # if world_rank == 0:
    #     import pdb;pdb.set_trace()

@distributed_setup
def test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/mixed_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_pre_checkpoint.items():
        if state_dict_pre_checkpoint[key].dtype == torch.float32:
            continue
        assert_allclose(value, state_dict_post_checkpoint[key])
    for key, value in state_dict_post_checkpoint.items():
        if state_dict_post_checkpoint[key].dtype == torch.float32:
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

@distributed_setup
def test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision(world_rank, world_size, device, checkpoint_dir = 'checkpoint_dir/distributed_zero/full_precision/'):
    learning_rate = 0.1
    seed = 1

    torch.manual_seed(seed)
    set_seed(seed)

    # PyTorch transformer model
    optim_config = optim.LambConfig(lr=learning_rate)
    opts = {
                'device' : {'id' : device},
                'mixed_precision':
                {
                    'enabled': True,
                    'loss_scaler': amp.DynamicLossScaler()
                },
                'distributed' :
                {
                    'world_rank' : world_rank,
                    'world_size' : world_size,
                    'allreduce_post_accumulation' : True,
                    'deepspeed_zero_optimization':
                    {
                        'stage': 1
                    }
                },
                'debug' : {'deterministic_compute': True}
            }
    opts = orttrainer.ORTTrainerOptions(opts)
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=opts)
    checkpoint.experimental_load_checkpoint(trainer, checkpoint_dir)

    # run an eval step to innitialize the graph
    torch.manual_seed(seed)
    set_seed(seed)
    data, targets = batcher_fn(train_data, 0)
    trainer.eval_step(data, targets)

    # load the saved state_dict to compare
    state = pickle.load(open(checkpoint_dir+'state_dict_'+str(world_rank)+'.pkl', 'rb'))

    # compare all states
    state_dict_pre_checkpoint = state['state_dict_'+str(world_rank)]
    state_dict_post_checkpoint = checkpoint.experimental_state_dict(trainer)
    for key, value in state_dict_post_checkpoint.items():
        if key.endswith('_fp16'):
            full_precision_key = key[:-5]
            assert full_precision_key in state_dict_pre_checkpoint
            assert_allclose(value, state_dict_pre_checkpoint[full_precision_key], atol=1e-3)
            continue
        assert_allclose(value, state_dict_pre_checkpoint[key])

if __name__ == '__main__':
    function_map = {
        'test_load_from_single_node_full_precision_into_single_node_full_precision': test_load_from_single_node_full_precision_into_single_node_full_precision,
        'test_load_from_single_node_mixed_precision_into_single_node_mixed_precision': test_load_from_single_node_mixed_precision_into_single_node_mixed_precision,
        'test_load_from_single_node_mixed_precision_into_single_node_full_precision': test_load_from_single_node_mixed_precision_into_single_node_full_precision,
        'test_load_from_single_node_full_precision_into_single_node_mixed_precision': test_load_from_single_node_full_precision_into_single_node_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_single_node_full_precision': test_load_from_data_parallelism_full_precision_into_single_node_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision': test_load_from_data_parallelism_mixed_precision_into_single_node_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_single_node_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision': test_load_from_data_parallelism_full_precision_into_single_node_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_single_node_full_precision': test_load_from_distributed_zero_full_precision_into_single_node_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision': test_load_from_distributed_zero_mixed_precision_into_single_node_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision': test_load_from_distributed_zero_full_precision_into_single_node_mixed_precision,

        'test_load_from_single_node_full_precision_into_data_parallelism_full_precision': test_load_from_single_node_full_precision_into_data_parallelism_full_precision,
        'test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision': test_load_from_single_node_mixed_precision_into_data_parallelism_full_precision,
        'test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_single_node_mixed_precision_into_data_parallelism_mixed_precision,
        'test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision': test_load_from_single_node_full_precision_into_data_parallelism_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision': test_load_from_data_parallelism_full_precision_into_data_parallelism_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision': test_load_from_data_parallelism_mixed_precision_into_data_parallelism_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_data_parallelism_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision': test_load_from_data_parallelism_full_precision_into_data_parallelism_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_full_precision_into_data_parallelism_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision': test_load_from_distributed_zero_mixed_precision_into_data_parallelism_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_data_parallelism_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision': test_load_from_distributed_zero_full_precision_into_data_parallelism_mixed_precision,

        'test_load_from_single_node_full_precision_into_distributed_zero_full_precision': test_load_from_single_node_full_precision_into_distributed_zero_full_precision,
        'test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_full_precision,
        'test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_single_node_mixed_precision_into_distributed_zero_mixed_precision,
        'test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision': test_load_from_single_node_full_precision_into_distributed_zero_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_full_precision,
        'test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_data_parallelism_mixed_precision_into_distributed_zero_mixed_precision,
        'test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision': test_load_from_data_parallelism_full_precision_into_distributed_zero_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_full_precision,
        'test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_mixed_precision_into_distributed_zero_mixed_precision,
        'test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision': test_load_from_distributed_zero_full_precision_into_distributed_zero_mixed_precision
    }
    # parser = argparse.ArgumentParser(description='Test saved states of trainers to loaded states')
    # parser.add_argument('scenario', choices=function_map.keys(), help='training scenario to test saved and loaded states')
    # args = parser.parse_args()
    # function_map[args.scenario]()
    test_load_from_distributed_zero_mixed_precision_into_single_node_mixed_precision()