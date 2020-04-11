import copy 

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box
import numpy as np
from collections import OrderedDict 
from dataclasses import dataclass

import world_of_supply_environment as ws

class WorldOfSupplyEnv(MultiAgentEnv):
    
    def __init__(self, env_config):
        self.episod_duration = env_config['episod_duration']
        self.reference_world = ws.WorldBuilder.create(80, 16)
        self.product_ids = self._product_ids()
        
        self.max_sources_per_facility = 0
        self.max_fleet_size = 0
        for f in self.reference_world.facilities.values():
            if f.consumer is not None:
                sources_num = len(f.consumer.sources)
                if sources_num > self.max_sources_per_facility:
                    self.max_sources_per_facility = sources_num
                    
            if f.distribution is not None:      
                if len(f.distribution.fleet) > self.max_fleet_size:
                    self.max_fleet_size = len(f.distribution.fleet)
                    
        
        self.action_space_dict = OrderedDict({ 
            'unit_price': (0, 1000),
            'end_unit_price': (0, 1000),
            'production_rate': (0, 1000),
            'product_id': (0, len(self.product_ids)),
            'source_id': (0, self.max_sources_per_facility),
            'consumer_quantity': (0, 10) })
             
        action_low  = [ dim[0] for dim in self.action_space_dict.values() ]
        action_high = [ dim[1] for dim in self.action_space_dict.values() ]
        self.action_space = Box(low=np.array(action_low), high=np.array(action_high), dtype=np.float32)
        
        
        example_state = self._world_to_state(self.reference_world)
        state_dim = len(list(example_state.values())[0])
        self.observation_space = Box(low=-1.0, high=2.0, shape=(state_dim, ), dtype=np.float32)

    def reset(self):
        self.world = copy.deepcopy(self.reference_world)
        self.time_step = 0
        return self._world_to_state(self.world)

    def step(self, action_dict):
        control = self._action_dictionary_to_control(action_dict)
        
        outcome = self.world.act(control)
        
        reward = WorldOfSupplyEnv._outcome_to_reward(outcome)
        state = self._world_to_state(self.world)
        
        self.time_step += 1
        is_done = self.time_step >= self.episod_duration
        done = { facility_id: is_done for facility_id in self.world.facilities.keys() }
        done['__all__'] = is_done
        
        return state, reward, done, {}
    
    def _product_ids(self):
        product_ids = set()
        for f in self.reference_world.facilities.values():
            product_ids.add(f.bom.output_product_id)
            product_ids.update(f.bom.inputs.keys())
        return list(product_ids)
    
    def _outcome_to_reward(outcome):
        return { f_id: sheet.total() for f_id, sheet in outcome.facility_step_balance_sheets.items() }
    
    def _action_dictionary_to_control(self, action_dict):
        controls = {}
        for facility_id, action in action_dict.items():
            controls[facility_id] = self._action_to_control(action, self.world.facilities[facility_id])
        return ws.World.Control(facility_controls = controls)
    
    def _action_to_control(self, action, facility):
        return ws.FacilityCell.Control(
                unit_price = self._float_to_int(action[0], 'unit_price'),
                end_unit_price = self._float_to_int(action[1], 'end_unit_price'),
                production_rate = self._float_to_int(action[2], 'production_rate'),
                consumer_product_id = self._float_to_product_id(action[3]),
                consumer_source_id = self._float_to_source_id(action[4], facility),
                consumer_quantity = self._float_to_int(action[5], 'consumer_quantity')
            )
    
    def _float_to_int(self, x, box_key):
        box = self.action_space_dict[box_key]
        return int( np.clip(round(x), a_min = box[0], a_max = box[1]) )
    
    def _float_to_product_id(self, x):
        pointer = int( np.clip(x, a_min = 0, a_max = len(self.product_ids) - 1) )
        return self.product_ids[ pointer ]
             
    def _float_to_source_id(self, x, facility):
        if facility.consumer is not None:
            sources_num = len(facility.consumer.sources)
            return int( np.clip(x, a_min = 0, a_max = sources_num - 1) )
        return 0
             
    def _world_to_state(self, world):
        state = {}
        for facility_id, facility in world.facilities.items():
            state[facility_id] = self._state(facility)    
        return self._serialize_state(state)  
    
    def _state(self, facility: ws.FacilityCell):
        state = {}
        
        state['balance'] = facility.economy.total_balance.total() / 1000
             
        state['bom_inputs'] = [0] * len(self.product_ids)  
        state['bom_outputs'] = [0] * len(self.product_ids) 
        for i, prod_id in enumerate(self.product_ids):
            if prod_id in facility.bom.inputs.keys():
                state['bom_inputs'][i] = facility.bom.inputs[prod_id]
            if prod_id == facility.bom.output_product_id:
                state['bom_outputs'][i] = facility.bom.output_lot_size
                
        state['fleet_position'] = [0] * self.max_fleet_size
        state['fleet_payload'] = [0] * self.max_fleet_size
        state['distributor_open_orders'] = 0
        state['distributor_open_orders_qty'] = 0
        if facility.distribution is not None:
            for i, v in enumerate(facility.distribution.fleet):
                state['fleet_position'][i] = WorldOfSupplyEnv._safe_div( v.location_pointer, v.path_len() )
                state['fleet_payload'][i] = v.payload
            
            q = facility.distribution.order_queue
            ordered_quantity = sum([ order.quantity for order in q ])
            state['distributor_open_orders'] = len(q)
            state['distributor_open_orders_qty'] = ordered_quantity
        
        # provide the agent with the statuses of tier-one suppliers' inventory and open orders
        state['consumer_source_inventory'] = [0] * ( len(self.product_ids) * self.max_sources_per_facility )
        state['consumer_open_orders'] = [0] * ( len(self.product_ids) * self.max_sources_per_facility )
        if facility.consumer is not None:
            for i_s, source in enumerate(facility.consumer.sources):
                for i_p, product_id in enumerate(self.product_ids):
                    i = i_s * self.max_sources_per_facility + i_p
                    state['consumer_source_inventory'][i] = source.storage.stock_levels[product_id]   
                    if facility.consumer.open_orders[source.id] is not None:
                        state['consumer_open_orders'][i] = facility.consumer.open_orders[source.id][product_id]
                        
        state['sold_units'] = 0
        if facility.seller is not None:
            state['sold_units'] = facility.seller.economy.total_units_sold
        
        state['storage_usage'] = WorldOfSupplyEnv._safe_div( facility.storage.used_capacity(), facility.storage.max_capacity )
        state['storage_levels'] = [0] * len(self.product_ids)     
        for i, prod_id in enumerate(self.product_ids):
            if prod_id in facility.storage.stock_levels.keys():
                 state['storage_levels'][i] = facility.storage.stock_levels[prod_id]
                    
        return state
    
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