import world_of_supply_rllib as wsr

import random

import ray
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_tf
import ray.rllib.agents.trainer_template as tt

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
import ray.rllib.agents.ddpg as ddpg


tf = try_import_tf()

ray.shutdown()
ray.init()

# Configuration ===============================================================================

env_config = {'episod_duration': 1000, 'global_reward_weight': 0.5}
env = wsr.WorldOfSupplyEnv(env_config)

base_trainer_config = {
    'env_config': env_config,
    'timesteps_per_iteration': 20000,
    
    # == Environment Settings == 
    'lr': 0.0001,
    'gamma': 0.95,
    
    # === Settings for the Trainer process ===
    'train_batch_size': 10000,
    'batch_mode': 'complete_episodes',
    #'rollout_fragment_length': 1000,
}


# Policy Configuration ===============================================================================

policies = {
        'baseline': (wsr.SimplePolicy, env.observation_space, env.action_space, {
            'facility_types': env.facility_types, 
            'number_of_products': env.n_products()
        }),
        'ppo': (PPOTFPolicy, env.observation_space, env.action_space, {
            # == LSTM ==
            "use_lstm": True,
            "max_seq_len": 500,
            "lstm_cell_size": 32,
            
            # === Model ===
            "model": {
               "fcnet_hiddens": [256, 256],
            }
        })
    }

def create_policy_mapping_fn(policy_names):
    # policy mapping is sampled once per episod, so we just randomly assing a policy to each agent_id
    return lambda agent_id: random.choice(policy_names)


# Training Routines ===============================================================================

def play_baseline(n_iterations):
    
    HandCodedTrainer = tt.build_trainer("HandCoded", wsr.SimplePolicy)
    ext_conf = {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": create_policy_mapping_fn(['baseline']),
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
    

def train_ppo(n_iterations, competing_policies = ['baseline', 'ppo']):
        
    ext_conf = {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": create_policy_mapping_fn(competing_policies),
                "policies_to_train": ['ppo']
            }
        }
    ppo_trainer = PPOTrainer(
        env = wsr.WorldOfSupplyEnv,
         config = dict(base_trainer_config, **ext_conf))
    
    for i in range(n_iterations):
        print("== Iteration", i, "==")
        print(pretty_print(ppo_trainer.train()))
        #checkpoint = ppo_trainer.save()
        #print("Checkpoint saved at", checkpoint)
        
    return ppo_trainer