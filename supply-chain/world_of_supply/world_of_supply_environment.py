from abc import ABC
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
import networkx as nx
from enum import Enum, auto

class Cell(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.x}, {self.y})"
    
class Agent(ABC):
    def act(self, control):
        pass
    
@dataclass
class BalanceSheet:
    profit: int = 0
    loss: int = 0
        
    def total(self) -> int:
        return self.profit + self.loss
        
    def __add__(self, other):
        return BalanceSheet(self.profit + other.profit, self.loss + other.loss)
    
    def __sub__(self, other):
        return BalanceSheet(self.profit - other.profit, self.loss - other.loss)
    
    def __repr__(self):
        return f"{self.profit+self.loss} ({self.profit} {self.loss})"
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    

# ======= Infrastructure 
class TerrainCell(Cell):
    def __init__(self, x, y):
        super(TerrainCell, self).__init__(x, y)

class RailroadCell(Cell):
    def __init__(self, x, y):
        super(RailroadCell, self).__init__(x, y)

        
# ======= Transportation
class Transport(Agent):
    
    @dataclass 
    class Economy:
        unit_transport_cost: int   # cost per unit per movement
        
        def step_balance_sheet(self, transport):
            return BalanceSheet(0, -transport.payload * self.unit_transport_cost * abs(transport.step))
    
    @dataclass
    class Control:
        pass
    
    def __init__(self, source, economy):
        self.source = source
        self.destination = None
        self.path = None
        self.location_pointer = 0
        self.step = 0
        self.payload = 0 # units
        self.economy = economy

    def schedule(self, world, destination, product_id, quantity):
        self.destination = destination
        self.product_id = product_id
        self.requested_quantity = quantity
        self.path = world.find_path(self.source.x, self.source.y, self.destination.x, self.destination.y)
        if self.path == None:
            raise Exception(f"Destination {destination} is unreachable")
        self.step = 1    # 1 - to destination, -1 - to source, 0 - finished
        self.location_pointer = 0

    def path_len(self):
        if self.path == None:
            return 0
        else:
            return len(self.path)
    
    def is_enroute(self):
        return self.destination is not None

    def current_location(self):
        if self.path == None:
            return (self.source.x, self.source.y)
        else:
            return self.path[self.location_pointer]
        
    def try_loading(self, quantity):
        if self.source.storage.try_take_units({ self.product_id: quantity }):
            self.payload = quantity
        
    def try_unloading(self):        
        unloaded = self.destination.storage.try_add_units({ self.product_id: self.payload }, all_or_nothing = False)
        if len(unloaded) > 0:
            unloaded_units = sum(unloaded.values())
            self.destination.consumer.on_order_reception(self.source.id, self.product_id, unloaded_units)
            self.payload = 0     # all units that were not sucessfully unloaded will be lost

    def act(self, control):
        if self.step > 0: 
            if self.location_pointer == 0 and self.payload == 0:
                self.try_loading(self.requested_quantity)
            
            if self.payload > 0:     # will stay at the source until loaded
                if self.location_pointer < len(self.path) - 1:
                    self.location_pointer += self.step
                else:
                    self.step = -1   # arrived to the destination

        if self.step < 0: 
            if self.location_pointer == len(self.path) - 1 and self.payload > 0:
                self.try_unloading()
                
            if self.payload == 0:    # will stay at the destination until unloaded
                if self.location_pointer > 0: 
                    self.location_pointer += self.step
                else:
                    self.step = 0    # arrived back to the source
                    self.destination = None
                    
        return self.economy.step_balance_sheet(self)


# ======= Basic facility components (units)

@dataclass    
class BillOfMaterials:
    # One manufacturing cycle consumes inputs 
    # and produces output_lot_size units of output_product_id
    inputs: Counter  # (product_id -> quantity per lot)
    output_product_id: str
    output_lot_size: int = 1
        
    def input_units_per_lot(self):
        return sum(self.inputs.values())
        

class StorageUnit(Agent):
    
    @dataclass 
    class Economy:
        unit_storage_cost: int    # cost per unit per time step
        
        def step_balance_sheet(self, storage):
            return BalanceSheet(0, -storage.used_capacity() * self.unit_storage_cost)
        
    @dataclass
    class Config:
        max_storage_capacity: int
        unit_storage_cost: int
    
    def __init__(self, max_capacity, economy):
        self.max_capacity = max_capacity
        self.stock_levels = Counter()
        self.economy = economy
    
    def used_capacity(self):
        return sum(self.stock_levels.values())
    
    def available_capacity(self):
        return self.max_capacity - self.used_capacity()
    
    def try_add_units(self, product_quantities, all_or_nothing = True) -> dict:
        
        # validation
        if all_or_nothing and self.available_capacity() < sum(product_quantities.values()):
            return {}
        
        # depositing
        unloaded_quantities = {}
        for p_id, q in product_quantities.items(): 
            unloading_qty = min(self.available_capacity(), q)
            self.stock_levels[p_id] += unloading_qty
            unloaded_quantities[p_id] = unloading_qty
            
        return unloaded_quantities
    
    def try_take_units(self, product_quantities):
        # validation
        for p_id, q in product_quantities.items():
            if self.stock_levels[p_id] < q:
                return False
        # withdrawal
        for p_id, q in product_quantities.items():
            self.stock_levels[p_id] -= q  
        return True
    
    def take_available(self, product_id, quantity):
        available = self.stock_levels[product_id]
        actual = min(available, quantity)
        self.stock_levels[product_id] -= actual
        return actual
    
    def act(self, control = None):
        return self.economy.step_balance_sheet(self)
        

class DistributionUnit(Agent):
    @dataclass 
    class Economy:
        unit_price: int = 0
        wrong_order_penatly: int = 0
        pending_order_penalty: int = 0
        order_checkin: int = 0  # balance for the current time step
            
        total_wrong_order_penalties: int = 0
        total_pending_order_penalties: int = 0
            
        def profit(self, units_sold):
            return self.unit_price * units_sold
        
    @dataclass
    class Config:  
        fleet_size: int
        unit_transport_cost: int
        wrong_order_penatly: int
        pending_order_penalty: int
    
    @dataclass
    class Control:
        unit_price: int                            
    
    @dataclass
    class Order:
        destination: Cell
        product_id: str
        quantity: int
        unit_price: int = 0
    
    def __init__(self, facility, fleet_size, distribution_economy, transport_economy):
        self.facility = facility
        self.fleet = [ Transport(facility, transport_economy) for i in range(fleet_size) ]
        self.order_queue = deque()
        self.economy = distribution_economy
        
    def place_order(self, order):
        if order.quantity > 0:
            if self.facility.bom.output_product_id == order.product_id:
                order.unit_price = self.economy.unit_price
                self.order_queue.append(order)   # add order to the queue
                order_total = self.economy.profit(order.quantity)
                self.economy.order_checkin += order_total
                return -order_total
            else:
                penalty = -self.economy.wrong_order_penatly * order.quantity
                self.economy.total_wrong_order_penalties += penalty
                return -self.economy.wrong_order_penatly * order.quantity
        else:
            return 0
            
    def act(self, control):
        if control is not None:
            self.economy.unit_price = control.unit_price    # update unit price
        
        transportation_balance = BalanceSheet()
        for vechicle in self.fleet:
            if len(self.order_queue) > 0 and not vechicle.is_enroute():
                order = self.order_queue.popleft()
                vechicle.schedule( self.facility.world, order.destination, order.product_id, order.quantity )
                transportation_balance -= vechicle.act(None)
            else:
                transportation_balance -= vechicle.act(None)
        
        pending_orders_loss = - self.economy.pending_order_penalty * len(self.order_queue)
        self.economy.total_pending_order_penalties += pending_orders_loss
        order_step_profit = self.economy.order_checkin
        self.economy.order_checkin = 0
        return BalanceSheet(order_step_profit, pending_orders_loss) + transportation_balance

    
class ManufacturingUnit(Agent):
    @dataclass 
    class Economy:
        unit_cost: int                   # production cost per unit 
            
        def cost(self, units_produced):
            return -self.unit_cost * units_produced
        
        def step_balance_sheet(self, units_produced):
            return BalanceSheet(0, self.cost(units_produced))
        
    @dataclass
    class Config:
        unit_manufacturing_cost: int
    
    @dataclass
    class Control:
        production_rate: int                  # lots per time step
    
    def __init__(self, facility, economy):
        self.facility = facility
        self.economy = economy
    
    def act(self, control):
        units_produced = 0
        
        if control is not None:
            for i in range(control.production_rate):
                # check we have enough storage space for the output lot
                if self.facility.storage.available_capacity() >= self.facility.bom.output_lot_size - self.facility.bom.input_units_per_lot(): 
                    # check we have enough input materials 
                    if self.facility.storage.try_take_units(self.facility.bom.inputs):                
                        self.facility.storage.stock_levels[self.facility.bom.output_product_id] += self.facility.bom.output_lot_size
                        units_produced += self.facility.bom.output_lot_size
                    
        return self.economy.step_balance_sheet(units_produced)  

    
class ConsumerUnit(Agent):
    
    @dataclass
    class Economy:
        total_units_purchased: int = 0
        total_units_received: int = 0
    
    @dataclass
    class Control:
        consumer_product_id: int  # what to purchase
        consumer_source_id: int   # where to purchase  
        consumer_quantity: int    # how many to purchase
            
    @dataclass
    class Config:
        sources: list
    
    def __init__(self, facility, sources):
        self.facility = facility
        self.sources = sources
        self.open_orders = {}
        self.economy = ConsumerUnit.Economy()
    
    def on_order_reception(self, source_id, product_id, quantity):
        self.economy.total_units_received += quantity
        self._update_open_orders(source_id, product_id, -quantity)
    
    def act(self, control):
        if control is None or control.consumer_product_id is None or control.consumer_quantity <= 0:
            return BalanceSheet()
            
        source_obj = self.sources[control.consumer_source_id]
        self._update_open_orders(source_obj.id, control.consumer_product_id, control.consumer_quantity)
        order = DistributionUnit.Order(self.facility, control.consumer_product_id, control.consumer_quantity)
        order_cost = source_obj.distribution.place_order( order )
        self.economy.total_units_purchased += control.consumer_quantity
        
        return BalanceSheet(0, order_cost)
    
    def _update_open_orders(self, source_id, product_id, qty_delta):
        if qty_delta > 0:
            if source_id not in self.open_orders:
                self.open_orders[source_id] = Counter()
            self.open_orders[source_id][product_id] += qty_delta
        else:
            self.open_orders[source_id][product_id] += qty_delta
            self.open_orders[source_id] += Counter() # remove zeros
            if len(self.open_orders[source_id]) == 0:
                del self.open_orders[source_id]
        
    
class SellerUnit(Agent):
    @dataclass
    class Economy:
        price_demand_intercept: int
        price_demand_slope: int
        unit_price: int = 0
        total_units_sold: int = 0
            
        def market_demand(self, unit_price):
            return max(0, self.price_demand_intercept - self.price_demand_slope * unit_price)
        
        def profit(self, units_sold, unit_price):
            return units_sold * unit_price
        
        def step_balance_sheet(self, units_sold, unit_price):
            return BalanceSheet(self.profit(units_sold, unit_price), 0)
        
    @dataclass
    class Config:
        price_demand_intercept: float
        price_demand_slope: float  
    
    @dataclass
    class Control:
        unit_price: int
            
    def __init__(self, facility, economy):
        self.facility = facility
        self.economy = economy 
            
    def act(self, control):
        if control is not None:
            self.economy.unit_price = control.unit_price   # update the current unit price
        
        product_id = self.facility.bom.output_product_id
        demand = self.economy.market_demand(self.economy.unit_price)
        sold_qty = self.facility.storage.take_available(product_id, demand)
        self.economy.total_units_sold += sold_qty
        return self.economy.step_balance_sheet( sold_qty, self.economy.unit_price )

# ======= Base facility class (collection of units)

class FacilityCell(Cell, Agent):
    @dataclass
    class Config(StorageUnit.Config, 
                 ConsumerUnit.Config, 
                 DistributionUnit.Config, 
                 ManufacturingUnit.Config, 
                 SellerUnit.Config):
        bill_of_materials: BillOfMaterials 
            
    @dataclass
    class EconomyConfig:
        initial_balance: int
    
    @dataclass 
    class Economy:
        total_balance: BalanceSheet
        
        def deposit(self, balance_sheets):
            total_balance_sheet = sum(balance_sheets)
            self.total_balance += total_balance_sheet
            return total_balance_sheet  
    
    @dataclass
    class Control(ConsumerUnit.Control,
                  DistributionUnit.Control, 
                  ManufacturingUnit.Control, 
                  SellerUnit.Control):
        pass
    
    def __init__(self, x, y, world, config, economy_config):  
        super(FacilityCell, self).__init__(x, y)
        self.id_num = world.generate_id()
        self.id = f"{self.__class__.__name__}_{self.id_num}"
        self.world = world
        self.economy = FacilityCell.Economy(BalanceSheet(economy_config.initial_balance, 0))
        self.bom = config.bill_of_materials
        self.storage = None
        self.consumer = None
        self.manufacturing = None
        self.distribution = None
        self.seller = None
    
    def act(self, control): 
        units = filter(None, [self.storage, self.consumer, self.manufacturing, self.distribution, self.seller])
        balance_sheets = [ u.act(control) for u in  units ]
        return self.economy.deposit(balance_sheets)


# ======= Concrete facility classes

def create_distribution_unit(facility, config):
    return DistributionUnit(facility, 
                            config.fleet_size, 
                            DistributionUnit.Economy(
                                wrong_order_penatly = config.wrong_order_penatly, 
                                pending_order_penalty = config.pending_order_penalty), 
                            Transport.Economy(config.unit_transport_cost))

class RawMaterialsFactoryCell(FacilityCell):
    def __init__(self, x, y, world, config, economy_config):
        super(RawMaterialsFactoryCell, self).__init__(x, y, world, config, economy_config) 
        self.storage = StorageUnit(config.max_storage_capacity, StorageUnit.Economy(config.unit_storage_cost))
        self.manufacturing = ManufacturingUnit(self, ManufacturingUnit.Economy(config.unit_manufacturing_cost))
        self.distribution = create_distribution_unit(self, config)
        
class SteelFactoryCell(RawMaterialsFactoryCell):
    def __init__(self, x, y, world, config, economy_config):
        super(SteelFactoryCell, self).__init__(x, y, world, config, economy_config)

class LumberFactoryCell(RawMaterialsFactoryCell):
    def __init__(self, x, y, world, config, economy_config):
        super(LumberFactoryCell, self).__init__(x, y, world, config, economy_config)

class ValueAddFactoryCell(FacilityCell):
    def __init__(self, x, y, world, config, economy_config):
        super(ValueAddFactoryCell, self).__init__(x, y, world, config, economy_config) 
        self.storage = StorageUnit(config.max_storage_capacity, StorageUnit.Economy(config.unit_storage_cost))
        self.consumer = ConsumerUnit(self, config.sources)
        self.manufacturing = ManufacturingUnit(self, ManufacturingUnit.Economy(config.unit_manufacturing_cost))
        self.distribution = create_distribution_unit(self, config)
    
class ToyFactoryCell(ValueAddFactoryCell):
    def __init__(self, x, y, world, config, economy_config):
        super(ToyFactoryCell, self).__init__(x, y, world, config, economy_config)

class WarehouseCell(FacilityCell):
    def __init__(self, x, y, world, config, economy_config):
        super(WarehouseCell, self).__init__(x, y, world, config, economy_config) 
        self.storage = StorageUnit(config.max_storage_capacity, StorageUnit.Economy(config.unit_storage_cost))
        self.consumer = ConsumerUnit(self, config.sources)
        self.distribution = create_distribution_unit(self, config)
        
class RetailerCell(FacilityCell):
    def __init__(self, x, y, world, config, economy_config):
        super(RetailerCell, self).__init__(x, y, world, config, economy_config) 
        self.storage = StorageUnit(config.max_storage_capacity, StorageUnit.Economy(config.unit_storage_cost))
        self.consumer = ConsumerUnit(self, config.sources)
        self.seller = SellerUnit(self, SellerUnit.Economy(config.price_demand_intercept, config.price_demand_slope))

        
# ======= The world

class World:
    
    @dataclass
    class Economy:
        def __init__(self, world):
            self.world = world
            
        def global_balance(self) -> BalanceSheet: 
            return sum([ f.economy.total_balance for f in self.world.facilities.values() ])  
            
    
    @dataclass
    class Control:
        facility_controls: dict
    
    @dataclass
    class StepOutcome:
        facility_step_balance_sheets: dict
    
    def __init__(self, x, y):
        self.size_x = x
        self.size_y = y
        self.grid = None
        self.economy = World.Economy(self)
        self.facilities = dict()
        self.id_counter = 0
        self.time_step = 0
        
    def generate_id(self):
        self.id_counter += 1
        return self.id_counter
        
    def act(self, control):
        balance_sheets = dict()
        for facility in self.facilities.values():
            balance_sheets[facility.id] = facility.act( control.facility_controls.get(facility.id) )
        
        self.time_step += 1
        return World.StepOutcome(balance_sheets)
    
    def create_cell(self, x, y, clazz):
        self.grid[x][y] = clazz(x, y)

    def place_cell(self, *cells):
        for c in cells:
            self.grid[c.x][c.y] = c
    
    def is_railroad(self, x, y):
        return isinstance(self.grid[x][y], RailroadCell)
    
    def is_traversable(self, x, y):
        return not isinstance(self.grid[x][y], TerrainCell)
    
    def c_tostring(x,y):
        return np.array([x,y]).tostring()
                
    def map_to_graph(self):
        g = nx.Graph()
        for x in range(1, self.size_x-1):
            for y in range(1, self.size_y-1):
                for c in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                    if self.is_traversable(x, y) and self.is_traversable(c[0], c[1]):
                        g.add_edge(World.c_tostring(x, y), World.c_tostring(c[0], c[1]))
        return g
    
    @lru_cache(maxsize = 32)  # speedup the simulation
    def find_path(self, x1, y1, x2, y2):
        g = self.map_to_graph()
        path = nx.astar_path(g, source=World.c_tostring(x1, y1), target=World.c_tostring(x2, y2))
        path_np = [np.fromstring(p, dtype=int) for p in path]
        return [(p[0], p[1]) for p in path_np]
    
    def get_facilities(self, clazz):
        return filter(lambda f: isinstance(f, clazz), self.facilities.values())
    
    
class WorldBuilder:
    def create(x = 80, y = 16):
        world = World(x, y)
        world.grid = [[TerrainCell(xi, yi) for yi in range(y)] for xi in range(x)]
        
        # parameters
        def default_facility_config(bom, sources):
            return FacilityCell.Config(bill_of_materials = bom, 
                                       max_storage_capacity = 20,
                                       unit_storage_cost = 1,
                                       fleet_size = 1,
                                       unit_transport_cost = 1,
                                       sources = sources,
                                       wrong_order_penatly = 500,
                                       pending_order_penalty = 4,
                                       unit_manufacturing_cost = 100,
                                       price_demand_intercept = 50,
                                       price_demand_slope = 0.005)
        
        def default_economy_config(initial_balance = 1000):
            return FacilityCell.EconomyConfig(initial_balance)
        
        steel_bom = BillOfMaterials(Counter(), 'steel', 1)
        lumber_bom = BillOfMaterials(Counter(), 'lumber', 1)
        toy_bom = BillOfMaterials(Counter({'lumber': 1, 'steel': 1}), 'toy_car')
        distribution_bom = BillOfMaterials(Counter({'toy_car': 1}), 'toy_car' )
        retailer_bom = BillOfMaterials(Counter({'toy_car': 1}), 'toy_car', 1)
        
        # facility placement
        map_margin = 2
        size_y_margins = world.size_y - 2*map_margin
        
        # raw materials
        steel_01 = SteelFactoryCell(10, 6, world, default_facility_config(steel_bom, None), default_economy_config() ) 
        lumber_01 = LumberFactoryCell(10, 10, world, default_facility_config(lumber_bom, None), default_economy_config() )
        raw_materials = [steel_01, lumber_01]
        world.place_cell(*raw_materials) 
        
        # manufacturing
        n_toy_factories = 3
        factories = []
        for i in range(n_toy_factories):
            f = ToyFactoryCell(35, int(size_y_margins/(n_toy_factories - 1)*i + map_margin), 
                                    world, default_facility_config(toy_bom, raw_materials), 
                                    default_economy_config() )
            world.place_cell(f) 
            factories.append(f)
            WorldBuilder.connect_cells(world, f, *raw_materials)
            
        # distribution  
        n_warehouses = 2
        warehouses = []
        for i in range(n_warehouses):
            w =  WarehouseCell(50, int(size_y_margins/(n_warehouses - 1)*i + map_margin), 
                               world, default_facility_config(distribution_bom, factories),
                               default_economy_config(2000) )
            world.place_cell(w) 
            warehouses.append(w)
            WorldBuilder.connect_cells(world, w, *factories)
            
        # final consumers
        n_retailers = 2
        retailers = []
        for i in range(n_retailers):
            retailer_config = default_facility_config(retailer_bom, warehouses)
            retailer_config.max_storage_capacity = 10
            r = RetailerCell(70, int(size_y_margins/(n_retailers - 1)*i + map_margin), 
                                  world, retailer_config, default_economy_config(3000) )
            world.place_cell(r)
            retailers.append(r)
            WorldBuilder.connect_cells(world, r, *warehouses)
    
        for facility in raw_materials + factories + warehouses + retailers:
            world.facilities[facility.id] = facility
        
        return world
        
    def connect_cells(world, source, *destinations):
        for dest_cell in destinations:
            WorldBuilder.build_railroad(world, source.x, source.y, dest_cell.x, dest_cell.y)
        
    def build_railroad(world, x1, y1, x2, y2):
        step_x = np.sign(x2 - x1)
        step_y = np.sign(y2 - y1)

        # make several attempts to find a route non-adjacent to existing roads  
        for i in range(5):
            xi = min(x1, x2) + int(abs(x2 - x1) * rnd.uniform(0.15, 0.85))
            if not (world.is_railroad(xi-1, y1+step_y) or world.is_railroad(xi+1, y1+step_y)):
                break

        for x in range(x1 + step_x, xi, step_x):
            world.create_cell(x, y1, RailroadCell) 
        if step_y != 0:
            for y in range(y1, y2, step_y):
                world.create_cell(xi, y, RailroadCell) 
        for x in range(xi, x2, step_x):
            world.create_cell(x, y2, RailroadCell) 


#  ======= Baseline control policies               
class SimpleControlPolicy:    
    
    def compute_control(self, world):
        
        def default_facility_control(unit_price, product_id, source_id):
            return FacilityCell.Control(
                unit_price = unit_price,
                production_rate = 4,
                consumer_product_id = product_id,
                consumer_source_id = source_id,
                consumer_quantity = 8
            )
        
        ctrl = dict()
        for f in world.get_facilities(RawMaterialsFactoryCell):
            ctrl[f.id] = default_facility_control(400, None, None)
        
        for f in world.get_facilities(ValueAddFactoryCell):
            ctrl[f.id] = default_facility_control(1000, *self._find_source(f))
            
        for f in world.get_facilities(WarehouseCell):
            ctrl[f.id] = default_facility_control(1400, *self._find_source(f))
            
        for f in world.get_facilities(RetailerCell):
            ctrl[f.id] = default_facility_control(1800, *self._find_source(f))
            
        return World.Control(ctrl)
    
    def _find_source(self, facility):
        # stop placing orders when the facility ran out of money
        if facility.economy.total_balance.total() <= 0: 
            return (None, None)
            
        inputs = facility.bom.inputs
        available_inventory = facility.storage.stock_levels
        in_transit_orders = sum(facility.consumer.open_orders.values(), Counter())
        booked_inventory = available_inventory + in_transit_orders
        
        # stop placing orders when the facilty runs out of capacity
        if sum(booked_inventory.values()) > facility.storage.max_capacity:
            return (None, None)
        
        most_neeed_product_id = None
        min_ratio = float('inf')
        for product_id, quantity in inputs.items():
            fulfillment_ratio = booked_inventory[product_id] / quantity
            if fulfillment_ratio < min_ratio:
                min_ratio = fulfillment_ratio
                most_neeed_product_id = product_id
        
        exporting_sources = SimpleControlPolicy.find_exporting_sources(facility, most_neeed_product_id)
                    
        return (most_neeed_product_id, rnd.choice(exporting_sources))
    
    def find_exporting_sources(facility, product_id):
        exporting_sources = []
        if product_id is not None and facility.consumer is not None:
            for source_id, source in enumerate(facility.consumer.sources):
                if source.bom.output_product_id == product_id:
                    exporting_sources.append(source_id)
        return exporting_sources
        