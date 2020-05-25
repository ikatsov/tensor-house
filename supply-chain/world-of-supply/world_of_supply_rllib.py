import copy 

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
import yaml

from gym.spaces import Box, Tuple, MultiDiscrete, Discrete
import numpy as np
from pprint import pprint
from collections import OrderedDict 
from dataclasses import dataclass
import random as rnd
import statistics
from itertools import chain
from collections import defaultdict

import world_of_supply_environment as ws

class Utils:
    def agentid_producer(facility_id):
        return facility_id + 'p'
    
    def agentid_consumer(facility_id):
        return facility_id + 'c'
    
    def is_producer_agent(agent_id):
        return agent_id[-1] == 'p'
    
    def is_consumer_agent(agent_id):
        return agent_id[-1] == 'c'
    
    def agentid_to_fid(agent_id):
        return agent_id[:-1]

class RewardCalculator:
    
    def __init__(self, env_config):
        self.env_config = env_config
    
    def calculate_reward(self, world, step_outcome) -> dict:
        return self._retailer_profit(world, step_outcome)
    
    def _retailer_profit(self, env, step_outcome):
        
        retailer_revenue = { f_id: sheet.profit for f_id, sheet in step_outcome.facility_step_balance_sheets.items() if "Retailer" in f_id}
        global_reward_retail_revenue = statistics.mean( retailer_revenue.values() )
        
        global_profits = { f_id: sheet.total() for f_id, sheet in step_outcome.facility_step_balance_sheets.items()}
        global_reward_total_profit = statistics.mean( global_profits.values() )
        
        w_gl_profit = 0.0 + 0.3 * (env.current_iteration / env.n_iterations)
        
        global_reward_producer = (1 - w_gl_profit) * global_reward_retail_revenue + w_gl_profit * global_reward_total_profit
        global_reward_consumer = (1 - w_gl_profit) * global_reward_retail_revenue + w_gl_profit * global_reward_total_profit
        
        wp = self.env_config['global_reward_weight_producer']
        wc = self.env_config['global_reward_weight_consumer']
        producer_reward_by_facility = { f_id: wp * global_reward_producer + (1 - wp) * sheet.total() for f_id, sheet in step_outcome.facility_step_balance_sheets.items() }
        consumer_reward_by_facility = { f_id: wc * global_reward_consumer + (1 - wc) * sheet.total() for f_id, sheet in step_outcome.facility_step_balance_sheets.items() }
        
        rewards_by_agent = {}
        for f_id, reward in producer_reward_by_facility.items():
            rewards_by_agent[Utils.agentid_producer(f_id)] = reward
        for f_id, reward in consumer_reward_by_facility.items():
            rewards_by_agent[Utils.agentid_consumer(f_id)] = reward
        return rewards_by_agent
    

class StateCalculator:
    
    def __init__(self, env):
        self.env = env
        
    def world_to_state(self, world):
        state = {}
        for facility_id, facility in world.facilities.items():
            f_state = self._state(facility)  
            self._add_global_features(f_state, world)
            state[Utils.agentid_producer(facility_id)] = f_state
            state[Utils.agentid_consumer(facility_id)] = f_state
            
        return self._serialize_state(state), state  
    
    def _state(self, facility: ws.FacilityCell):
        state = {}
        
        facility_type = [0] * len(self.env.facility_types)
        facility_type[self.env.facility_types[facility.__class__.__name__]] = 1
        state['facility_type'] = facility_type
        
        facility_id_one_hot = [0] * len(self.env.reference_world.facilities)
        facility_id_one_hot[facility.id_num - 1] = 1
        state['facility_id'] = facility_id_one_hot
        
        #state['balance_profit'] = self._balance_norm( facility.economy.total_balance.profit )
        #state['balance_loss'] = self._balance_norm( facility.economy.total_balance.loss )
        state['is_positive_balance'] = 1 if facility.economy.total_balance.total() > 0 else 0
             
        self._add_bom_features(state, facility)
        self._add_distributor_features(state, facility)
        self._add_consumer_features(state, facility)
                        
        #state['sold_units'] = 0
        #if facility.seller is not None:
        #    state['sold_units'] = facility.seller.economy.total_units_sold
        
        state['storage_capacity'] = facility.storage.max_capacity
        state['storage_levels'] = [0] * self.env.n_products()     
        for i, prod_id in enumerate(self.env.product_ids):
            if prod_id in facility.storage.stock_levels.keys():
                 state['storage_levels'][i] = facility.storage.stock_levels[prod_id]
        state['storage_utilization'] = sum(state['storage_levels']) / state['storage_capacity']
                    
        return state
    
    def _add_global_features(self, state, world):
        state['global_time'] = world.time_step / self.env.env_config['episod_duration']
        state['global_storage_utilization'] = [ f.storage.used_capacity() / f.storage.max_capacity for f in world.facilities.values() ]
        #state['balances'] = [ self._balance_norm(f.economy.total_balance.total()) for f in world.facilities.values() ]
        
    def _balance_norm(self, v):
        return v/1000
    
    def _add_bom_features(self, state, facility: ws.FacilityCell):
        state['bom_inputs'] = [0] * self.env.n_products()  
        state['bom_outputs'] = [0] * self.env.n_products() 
        for i, prod_id in enumerate(self.env.product_ids):
            if prod_id in facility.bom.inputs.keys():
                state['bom_inputs'][i] = facility.bom.inputs[prod_id]
            if prod_id == facility.bom.output_product_id:
                state['bom_outputs'][i] = facility.bom.output_lot_size
                
    def _add_distributor_features(self, state, facility: ws.FacilityCell):
        ##state['fleet_position'] = [0] * self.max_fleet_size
        ##state['fleet_payload'] = [0] * self.max_fleet_size
        state['distributor_in_transit_orders'] = 0
        state['distributor_in_transit_orders_qty'] = 0
        if facility.distribution is not None:
            ##for i, v in enumerate(facility.distribution.fleet):
            ##    state['fleet_position'][i] = WorldOfSupplyEnv._safe_div( v.location_pointer, v.path_len() )
            ##    state['fleet_payload'][i] = v.payload
            
            q = facility.distribution.order_queue
            ordered_quantity = sum([ order.quantity for order in q ])
            state['distributor_in_transit_orders'] = len(q)
            state['distributor_in_transit_orders_qty'] = ordered_quantity
            
    def _add_consumer_features(self, state, facility: ws.FacilityCell):
        # which source exports which product
        state['consumer_source_export_mask'] = [0] * ( self.env.n_products() * self.env.max_sources_per_facility )
        # provide the agent with the statuses of tier-one suppliers' inventory and in-transit orders
        ##state['consumer_source_inventory'] = [0] * ( self.env.n_products() * self.max_sources_per_facility )
        state['consumer_in_transit_orders'] = [0] * ( self.env.n_products() * self.env.max_sources_per_facility )
        if facility.consumer is not None:
            for i_s, source in enumerate(facility.consumer.sources):
                for i_p, product_id in enumerate(self.env.product_ids):
                    i = i_s * self.env.max_sources_per_facility + i_p
                    #state['consumer_source_inventory'][i] = source.storage.stock_levels[product_id]  
                    if source.bom.output_product_id == product_id:
                        state['consumer_source_export_mask'][i] = 1
                    if source.id in facility.consumer.open_orders:
                        state['consumer_in_transit_orders'][i] = facility.consumer.open_orders[source.id][product_id]
        
    def _safe_div(x, y):
        if y is not 0:
            return x
        return 0
    
    def _serialize_state(self, state):
        result = {}
        for k, v in state.items():
            state_vec = np.hstack(list(v.values()))
            state_normal = ( state_vec - np.min(state_vec) )  /  ( np.max(state_vec) - np.min(state_vec) )
            result[k] = state_normal
        return result


class ActionCalculator:
    
    def __init__(self, env):
        self.env = env
    
    def action_dictionary_to_control(self, action_dict, world):
        actions_by_facility = defaultdict(list)
        for agent_id, action in action_dict.items():
            f_id = Utils.agentid_to_fid(agent_id)
            actions_by_facility[f_id].append((agent_id, action)) 
        
        controls = {}
        for f_id, actions in actions_by_facility.items():
            controls[f_id] = self._actions_to_control( world.facilities[ f_id ], actions )
            
        return ws.World.Control(facility_controls = controls)
    
    
    def _actions_to_control(self, facility, actions):       
        unit_price_mapping = {
            0: 400,
            1: 600,
            2: 1000,
            3: 1200,
            4: 1400,
            5: 1600,
            6: 1800,
            7: 2000
        }
        
        small_controls_mapping = {
            0: 0,
            1: 2,
            2: 4,
            3: 6,
            4: 8,
            5: 10
        }
        
        def get_or_zero(arr, i):
            if i < len(arr):
                return arr[i]
            else:
                return 0
        
        n_facility_sources = len(facility.consumer.sources) if facility.consumer is not None else 0
        
        control = ws.FacilityCell.Control(
            unit_price = 0,
            production_rate = 0,
            consumer_product_id = 0,
            consumer_source_id = 0,
            consumer_quantity = 0   
        ) 
        for agent_id, action in actions:
            action = np.array(action).flatten()
        
            if Utils.is_producer_agent(agent_id):
                control.unit_price = unit_price_mapping[ get_or_zero(action, 0) ]
                control.production_rate = small_controls_mapping[ get_or_zero(action, 1) ]
                
            if Utils.is_consumer_agent(agent_id):
                
                product_id = self.env.product_ids[ get_or_zero(action, 0) ]
                exporting_sources = ws.SimpleControlPolicy.find_exporting_sources(facility, product_id)
                source_id_auto = 0 if len(exporting_sources)==0 else rnd.choice( exporting_sources )
                
                #source_id_policy = int( round(get_or_zero(action, 1) * (n_facility_sources-1) / self.env.max_sources_per_facility) )
                source_id_policy = min(get_or_zero(action, 1), n_facility_sources-1)
                
                control.consumer_product_id = product_id                    
                control.consumer_source_id =  source_id_policy   # source_id_auto
                control.consumer_quantity = small_controls_mapping[ get_or_zero(action, 2) ]
        
        return control

        

class WorldOfSupplyEnv(MultiAgentEnv):
    
    def __init__(self, env_config):
        self.env_config = env_config
        self.reference_world = ws.WorldBuilder.create()
        self.current_iteration = 0
        self.n_iterations = 0
        
        self.product_ids = self._product_ids()
        self.max_sources_per_facility = 0
        self.max_fleet_size = 0
        self.facility_types = {}
        facility_class_id = 0
        for f in self.reference_world.facilities.values():
            if f.consumer is not None:
                sources_num = len(f.consumer.sources)
                if sources_num > self.max_sources_per_facility:
                    self.max_sources_per_facility = sources_num
                    
            if f.distribution is not None:      
                if len(f.distribution.fleet) > self.max_fleet_size:
                    self.max_fleet_size = len(f.distribution.fleet)
                    
            facility_class = f.__class__.__name__
            if facility_class not in self.facility_types:
                self.facility_types[facility_class] = facility_class_id
                facility_class_id += 1
                
        self.state_calculator = StateCalculator(self)
        self.reward_calculator = RewardCalculator(env_config)
        self.action_calculator = ActionCalculator(self)
                         
        self.action_space_producer = MultiDiscrete([ 
            8,                             # unit price
            6,                             # production rate level
        ])
        
        self.action_space_consumer = MultiDiscrete([ 
            self.n_products(),                           # consumer product id
            self.max_sources_per_facility,               # consumer source id
            6                                            # consumer_quantity
        ])
                
        example_state, _ = self.state_calculator.world_to_state(self.reference_world)
        state_dim = len(list(example_state.values())[0])
        self.observation_space = Box(low=0.00, high=1.00, shape=(state_dim, ), dtype=np.float64)

    def reset(self):
        self.world = ws.WorldBuilder.create(80, 16)
        self.time_step = 0
        state, _ = self.state_calculator.world_to_state(self.world)
        return state

    def step(self, action_dict):
        control = self.action_calculator.action_dictionary_to_control(action_dict, self.world)
        
        outcome = self.world.act(control)
        self.time_step += 1
        
        # churn through no-action cycles 
        for i in range(self.env_config['downsampling_rate'] - 1): 
            nop_outcome = self.world.act(ws.World.Control({}))
            self.time_step += 1
            
            balances = outcome.facility_step_balance_sheets
            for agent_id in balances.keys():
                balances[agent_id] = balances[agent_id] + nop_outcome.facility_step_balance_sheets[agent_id]
            
        rewards = self.reward_calculator.calculate_reward(self, outcome)
        
        seralized_states, info_states = self.state_calculator.world_to_state(self.world)
        
        is_done = self.time_step >= self.env_config['episod_duration']
        dones = { agent_id: is_done for agent_id in seralized_states.keys() }
        dones['__all__'] = is_done
        
        return seralized_states, rewards, dones, info_states
    
    def agent_ids(self):
        agents = []
        for f_id in self.world.facilities.keys():
            agents.append(Utils.agentid_producer(f_id))
        for f_id in self.world.facilities.keys():
            agents.append(Utils.agentid_consumer(f_id))
        return agents
    
    def set_iteration(self, iteration, n_iterations):
        self.current_iteration = iteration
        self.n_iterations = n_iterations
    
    def n_products(self):
        return len(self._product_ids())
    
    def _product_ids(self):
        product_ids = set()
        for f in self.reference_world.facilities.values():
            product_ids.add(f.bom.output_product_id)
            product_ids.update(f.bom.inputs.keys())
        return list(product_ids)
    
    
    
class SimplePolicy(Policy):   
    
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.action_space_shape = action_space.shape
        self.n_products = config['number_of_products']
        self.n_sources = config['number_of_sources']

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs): 
        
        if info_batch is None:
            action_dict = [ self._action(f_state, None) for f_state in obs_batch ], [], {}  
        else:    
            action_dict = [self._action(f_state, f_state_info) for f_state, f_state_info in zip(obs_batch, info_batch)], [], {}
            
        return action_dict
    
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
    
    def get_config_from_env(env):
        return {'facility_types': env.facility_types, 
                'number_of_products': env.n_products(),
                'number_of_sources': env.max_sources_per_facility}
    
    
class ProducerSimplePolicy(SimplePolicy):
    
    def __init__(self, observation_space, action_space, config):
        SimplePolicy.__init__(self, observation_space, action_space, config)
        
        facility_types = config['facility_types']
        self.unit_prices = [0] * len(facility_types)
        unit_price_map = {
            ws.SteelFactoryCell.__name__: 0,  # $400
            ws.LumberFactoryCell.__name__: 0, # $400
            ws.ToyFactoryCell.__name__: 2,    # $1000
            ws.WarehouseCell.__name__: 4,     # $1400
            ws.RetailerCell.__name__: 6       # $1800
        }

        for f_class, f_id in facility_types.items():
            self.unit_prices[ f_id ] = unit_price_map[f_class]
            
    def _action(self, facility_state, facility_state_info):
        def default_facility_control(unit_price): 
            control = [
                unit_price,    # unit_price
                2,             # production_rate (level 2 -> 4 units)
            ]
            return control
                   
        action = default_facility_control(0)
        if facility_state_info is not None and len(facility_state_info) > 0:   
            unit_price = self.unit_prices[ np.flatnonzero( facility_state_info['facility_type'] )[0] ]
            action = default_facility_control(unit_price)
        
        return action
    
    
class ConsumerSimplePolicy(SimplePolicy):
    
    def __init__(self, observation_space, action_space, config):
        SimplePolicy.__init__(self, observation_space, action_space, config)
            
    def _action(self, facility_state, facility_state_info):
        def default_facility_control(source_id, product_id, order_qty = 4):  # (level 4 -> 8 units)
            control = [
                product_id,
                source_id,
                order_qty
            ]
            return control
                   
        action = default_facility_control(0, 0, 0)
        if facility_state_info is not None and len(facility_state_info) > 0:
            if np.count_nonzero(facility_state_info['bom_inputs']) > 0:
                action = default_facility_control(*self._find_source(facility_state_info))
            else:
                action = default_facility_control(0, 0, 0)
        
        return action
    
    def _find_source(self, f_state_info):
        # stop placing orders when the facility ran out of money
        #if f_state_info['balance_profit'] - f_state_info['balance_loss'] <= 0: 
        if f_state_info['is_positive_balance'] <= 0:
            return (0, 0, 0)
            
        inputs = f_state_info['bom_inputs']
        available_inventory = f_state_info['storage_levels']
        inflight_orders = np.sum(np.reshape(f_state_info['consumer_in_transit_orders'], (self.n_products, -1)), axis=0)
        booked_inventory = available_inventory + inflight_orders
        
        # stop placing orders when the facilty runs out of capacity
        if sum(booked_inventory) > f_state_info['storage_capacity']:
            return (0, 0, 0)
        
        most_neeed_product_id = None
        min_ratio = float('inf')
        for product_id, quantity in enumerate(inputs):
            if quantity > 0:
                fulfillment_ratio = booked_inventory[product_id] / quantity
                if fulfillment_ratio < min_ratio:
                    min_ratio = fulfillment_ratio
                    most_neeed_product_id = product_id
        
        exporting_sources = []
        if most_neeed_product_id is not None:
            for i in range(self.n_sources):
                if f_state_info['consumer_source_export_mask'][i*self.n_sources + most_neeed_product_id] == 1:
                    exporting_sources.append(i) 
                    
        return (rnd.choice(exporting_sources), most_neeed_product_id)