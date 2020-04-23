import world_of_supply_rllib as wsr

import random
import numpy as np

import ray
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_tf
import ray.rllib.agents.trainer_template as tt

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy


tf = try_import_tf()

ray.shutdown()
ray.init()

# Configuration ===============================================================================

env_config = {
    'episod_duration': 1000, 
    'global_reward_weight': 0.00
}
env = wsr.WorldOfSupplyEnv(env_config)

base_trainer_config = {
    'env_config': env_config,
    'timesteps_per_iteration': 20000,
    
    # == Environment Settings == 
    #'lr': 0.0005,
    'gamma': 0.99,
    
    # === Settings for the Trainer process ===
    'train_batch_size': 10000,
    'batch_mode': 'complete_episodes',
    #'rollout_fragment_length': 1000,
}


# Policy Configuration ===============================================================================

policies = {
        'baseline': (wsr.SimplePolicy, env.observation_space, env.action_space, wsr.SimplePolicy.get_config_from_env(env)),
        'ppo': (PPOTFPolicy, env.observation_space, env.action_space, {
            # === Model ===
            "model": {
                "fcnet_hiddens": [256, 256],
                #"fcnet_activation": "relu",
                
                # == LSTM ==
                "use_lstm": False,
                "max_seq_len": 128,
                "lstm_cell_size": 64, 
                "lstm_use_prev_action_reward": False,
            }
        })
    }

def filter_keys(d, keys):
    return {k:v for k,v in d.items() if k in keys}

policy_mapping_global = {
    'SteelFactory': 'baseline',
    'LumberFactory': 'baseline',
    'ToyFactory': 'ppo',
    'Warehouse': 'baseline',
    'Retailer': 'baseline'
}

def create_policy_mapping_fn(policy_map):
    # policy mapping is sampled once per episod
    def mapping_fn(agent_id):
        for f_filter, policy_name in policy_map.items():
            if f_filter in agent_id:
                return policy_name
        
    return mapping_fn
    


# Training Routines ===============================================================================

def print_training_results(result):
    keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        print(f"- {k}: {result[k]}")

def play_baseline(n_iterations):
    
    HandCodedTrainer = tt.build_trainer("HandCoded", wsr.SimplePolicy)
    ext_conf = {
            "multiagent": {
                "policies": filter_keys(policies, ['baseline']),
                "policy_mapping_fn": lambda agent_id: 'baseline',
                "policies_to_train": ['baseline']
            }
        }
    handcoded_trainer = HandCodedTrainer(
        env = wsr.WorldOfSupplyEnv,
        config = dict(base_trainer_config, **ext_conf))

    for i in range(n_iterations):
        print("== Iteration", i, "==")
        print(pretty_print(handcoded_trainer.train()))
        
    return handcoded_trainer
    

def train_ppo(n_iterations):
    
    policy_map = policy_mapping_global.copy()
    ext_conf = ppo.DEFAULT_CONFIG.copy()
    ext_conf.update({
            #"vf_clip_param": 1000.0,
            #"lr_schedule": [[0, 0.01], [100000, 0.0001]],
            "vf_share_layers": True,
            "vf_loss_coeff": 100.00,
            "multiagent": {
                "policies": filter_keys(policies, set(policy_mapping_global.values())),
                "policy_mapping_fn": create_policy_mapping_fn(policy_map),
                "policies_to_train": ['ppo']
            }
        })
    ppo_trainer = ppo.PPOTrainer(
        env = wsr.WorldOfSupplyEnv,
        config = dict(ext_conf, **base_trainer_config))
    
    for i in range(n_iterations):
        print(f"\n== Iteration {i} ==")
        
        #if i == int(0.5*n_iterations):
        #    print("== Enabling PPO for Lumber Factory")
        #    policy_map['LumberFactory'] = 'ppo'
        
        result = ppo_trainer.train()
        print_training_results(result)
        
    return ppo_trainer