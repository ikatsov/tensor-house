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

import world_of_supply_environment as ws

class RewardCalculator:
    
    def __init__(self, env_config):
        self.env_config = env_config
    
    def calculate_reward(self, world, step_outcome):
        return self._weighted_profit(world, step_outcome)
    
    def _retailer_profit(self, world, step_outcome):
        retailer_revenue = { f_id: sheet.profit for f_id, sheet in step_outcome.facility_step_balance_sheets.items() if "Retailer" in f_id}
        global_reward = statistics.mean( retailer_revenue.values() )
        w = self.env_config['global_reward_weight']
        return { f_id: w * global_reward + (1 - w) * sheet.total() for f_id, sheet in step_outcome.facility_step_balance_sheets.items() }
    
    def _weighted_profit(self, world, step_outcome):
        totals = { f_id: sheet.total() for f_id, sheet in step_outcome.facility_step_balance_sheets.items()}
        mean_total = statistics.mean( totals.values() )
        w = self.env_config['global_reward_weight']
        return { f_id: w * mean_total + (1 - w) * total for f_id, total in totals.items() }
    
    def _inventory_movement(self, world, step_outcome):
        old_inventory = self.inventory_per_facility
        self.inventory_per_facility = self._get_inventory_per_facility(self.world)
        inventory_movement = {k: abs(old_inventory[k] - self.inventory_per_facility[k]) for k in self.inventory_per_facility.keys() }
        return inventory_movement
    
    def _transport_movement(self, world, step_outcome):
        old_pointers = self.transport_pointers
        self.transport_pointers = self._get_transport_pointers(self.world)
        pointers_movement = {k: abs(old_pointers[k] - self.transport_pointers[k]) for k in self.transport_pointers.keys() }
        return pointers_movement
        

class WorldOfSupplyEnv(MultiAgentEnv):
    
    def __init__(self, env_config):
        self.env_config = env_config
        self.reference_world = ws.WorldBuilder.create(80, 16)
        self.product_ids = self._product_ids()
        self.reward_calculator = RewardCalculator(env_config)
        
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
                         
        self.action_space = Tuple([ 
            Discrete(8), # unit price increment/decrement level
            Discrete(5), # production rate level
            #Box(low=0, high=2000, shape=(1,), dtype=np.int64),
            #Box(low=0, high=20, shape=(1,), dtype=np.int64),
            Discrete(self.n_products()), # consumer product id
            Discrete(self.max_sources_per_facility), # consumer source id
            Discrete(5)  # consumer_quantity
        ])
                
        example_state, _ = self._world_to_state(self.reference_world)
        state_dim = len(list(example_state.values())[0])
        self.observation_space = Box(low=-1.0, high=1.0, shape=(state_dim, ), dtype=np.float32)
        
        status = ["Inialized WorldOfSupply", [f"Action space: {self.action_space}", f"Observation space: {self.observation_space}"] ]
        print(yaml.dump(status))

    def reset(self):
        self.world = copy.deepcopy(self.reference_world)
        self.time_step = 0
        self.inventory_per_facility = self._get_inventory_per_facility(self.world)
        self.transport_pointers = self._get_transport_pointers(self.world)
        state, _ = self._world_to_state(self.world)
        return state

    def step(self, action_dict):
        control = self._action_dictionary_to_control(action_dict)
        
        outcome = self.world.act(control)
        
        reward = self.reward_calculator.calculate_reward(self.world, outcome)
        seralized_state, info_state = self._world_to_state(self.world)
        
        self.time_step += 1
        is_done = self.time_step >= self.env_config['episod_duration']
        done = { facility_id: is_done for facility_id in self.world.facilities.keys() }
        done['__all__'] = is_done
        
        return seralized_state, reward, done, info_state
    
    def _get_inventory_per_facility(self, world):
        units_per_facility = {}
        for facility in world.facilities.values():
            if facility.storage is not None:
                units_per_facility[facility.id] = sum(facility.storage.stock_levels.values())
        return units_per_facility
    
    def _get_transport_pointers(self, world):
        pointers = {}
        for facility in world.facilities.values():
            if facility.distribution is not None:
                for t in facility.distribution.fleet:
                    pointers[facility.id] = t.location_pointer
            else:
                pointers[facility.id] = 0
        return pointers
    
    def n_products(self):
        return len(self._product_ids())
    
    def _product_ids(self):
        product_ids = set()
        for f in self.reference_world.facilities.values():
            product_ids.add(f.bom.output_product_id)
            product_ids.update(f.bom.inputs.keys())
        return list(product_ids)
    
    def _action_dictionary_to_control(self, action_dict):
        controls = {}
        for facility_id, action in action_dict.items():
            controls[facility_id] = self._action_to_control(action, self.world.facilities[facility_id])
        return ws.World.Control(facility_controls = controls)
    
    def _action_to_control(self, action, facility):
                 
        unit_price_mapping = {
            0: 400, 
            1: 600,
            2: 800,
            3: 1000,
            4: 1200,
            5: 1400,
            6: 1600,
            7: 1800
        }
        
        small_controls_mapping = {
            0: 0,
            1: 2,
            2: 4,
            3: 6,
            4: 8
        }
        
        n_facility_sources = len(facility.consumer.sources) if facility.consumer is not None else 0
        
        action = np.array(action).flatten()
        
        return ws.FacilityCell.Control(
                unit_price = unit_price_mapping[ action[0] ],
                production_rate = small_controls_mapping[ action[1] ],   
                #unit_price = int(action[0]) ,
                #production_rate = int(action[1]) ,
                consumer_product_id = self.product_ids[ action[2] ],                    
                consumer_source_id = min(action[3], n_facility_sources-1),
                consumer_quantity = small_controls_mapping[ action[4] ]
            )
    
    def _float_to_int(self, x, box_key):
        box = self.action_space_boxes[box_key]
        return int( np.clip(round(x), a_min = box[0], a_max = box[1]) )
             
    def _world_to_state(self, world):
        state = {}
        for facility_id, facility in world.facilities.items():
            state[facility_id] = self._state(facility)  
            self._add_global_features(state[facility_id], world)
            
        return self._serialize_state(state), state  
    
    def _state(self, facility: ws.FacilityCell):
        state = {}
        
        facility_type = [0] * len(self.facility_types)
        facility_type[self.facility_types[facility.__class__.__name__]] = 1
        state['facility_type'] = facility_type
        
        state['balance'] = self._balance_norm( facility.economy.total_balance.total() )
             
        self._add_bom_features(state, facility)
        ##self._add_distributor_features(state, facility)
        self._add_consumer_features(state, facility)
                        
        state['sold_units'] = 0
        if facility.seller is not None:
            state['sold_units'] = facility.seller.economy.total_units_sold
        
        state['storage_capacity'] = facility.storage.max_capacity
        state['storage_levels'] = [0] * len(self.product_ids)     
        for i, prod_id in enumerate(self.product_ids):
            if prod_id in facility.storage.stock_levels.keys():
                 state['storage_levels'][i] = facility.storage.stock_levels[prod_id]
                    
        return state
    
    def _add_global_features(self, state, world):
        state['time'] = world.time_step
        state['balances'] = [ self._balance_norm(f.economy.total_balance.total()) for f in world.facilities.values() ]
        
    def _balance_norm(self, v):
        return v/1000
    
    def _add_bom_features(self, state, facility: ws.FacilityCell):
        state['bom_inputs'] = [0] * len(self.product_ids)  
        state['bom_outputs'] = [0] * len(self.product_ids) 
        for i, prod_id in enumerate(self.product_ids):
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
        state['consumer_source_export_mask'] = [0] * ( len(self.product_ids) * self.max_sources_per_facility )
        # provide the agent with the statuses of tier-one suppliers' inventory and in-transit orders
        ##state['consumer_source_inventory'] = [0] * ( len(self.product_ids) * self.max_sources_per_facility )
        state['consumer_in_transit_orders'] = [0] * ( len(self.product_ids) * self.max_sources_per_facility )
        if facility.consumer is not None:
            for i_s, source in enumerate(facility.consumer.sources):
                for i_p, product_id in enumerate(self.product_ids):
                    i = i_s * self.max_sources_per_facility + i_p
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
    
    
class SimplePolicy(Policy):   
    
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.action_space_shape = action_space.shape
        
        facility_types = config['facility_types']
        self.unit_prices = [0] * len(facility_types)
        unit_price_map = {
            ws.SteelFactoryCell.__name__: 0, #400
            ws.LumberFactoryCell.__name__: 0, #400
            ws.ToyFactoryCell.__name__: 3, #1000
            ws.WarehouseCell.__name__: 4, # 1200
            ws.RetailerCell.__name__: 6 #1600
        }

        for f_class, f_id in facility_types.items():
            self.unit_prices[ f_id ] = unit_price_map[f_class]
        
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
            return [ self._action(f_state, None) for f_state in obs_batch ], [], {}
        
        return [self._action(f_state, f_state_info) for f_state, f_state_info in zip(obs_batch, info_batch)], [], {}
    
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
        
    def _action(self, facility_state, facility_state_info):
        def default_facility_control(unit_price, source_id, product_id, order_qty = 2):
            control = [
                unit_price,    # unit_price
                2,             # production_rate
                product_id,
                source_id,
                order_qty
            ]
            return control
                   
        action = default_facility_control(0, 0, 0, 0)
        if facility_state_info is not None and len(facility_state_info) > 0:   
            unit_price = self.unit_prices[ np.flatnonzero( facility_state_info['facility_type'] )[0] ]
            if np.count_nonzero(facility_state_info['bom_inputs']) > 0:
                action = default_facility_control(unit_price, *self._find_source(facility_state_info))
            else:
                action = default_facility_control(unit_price, 0, 0, 0)
        
        return action
    
    def _find_source(self, f_state_info):
        # stop placing orders when the facility ran out of money
        if f_state_info['balance'] <= 0: 
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