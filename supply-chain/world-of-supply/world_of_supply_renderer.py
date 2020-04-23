import PIL
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import Counter
import yaml
from multipledispatch import dispatch
import importlib
import math
import sys
from IPython.display import display, HTML

import world_of_supply_environment as ws

class Utils:
    def ascii_progress_bar(done, limit, bar_lenght_char = 15):
        if limit == 0:
            done_chars = 0
        else:
            done_chars = round(min(done, limit)/limit*bar_lenght_char)
        bar = ['='] * (done_chars)
        return ''.join(bar + (['-'] * (bar_lenght_char - done_chars)) + [f" {done}/{limit}"])

class WorldRenderer:
    def plot_sequence_images(image_array):
        ''' Display images sequence as an animation in jupyter notebook
        Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
        '''
        dpi = 72.0
        xpixels, ypixels = image_array[0].shape[:2]
        fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
        im = plt.figimage(image_array[0])

        def animate(i):
            im.set_array(image_array[i])
            return (im,)

        anim = animation.FuncAnimation(fig, animate, frames=len(image_array), interval=200, repeat_delay=1, repeat=True)
        display(HTML(anim.to_html5_video()))
        

class AsciiWorldStatusPrinter():
    
    @dispatch(ws.World)
    def status(world: ws.World) -> list:
        status = [ ["World:", [f"Time step: {world.time_step}", f"Global balance: {world.economy.global_balance()}"]] ]
        for f in world.facilities.values():
            status.append(AsciiWorldStatusPrinter.status(f))    
            
        return status  
    
    def cell_status(cell):
        return [f"{cell.__class__.__name__} ({cell.x}, {cell.y})"]    
    
    @dispatch(ws.FacilityCell)
    def status(facility: ws.FacilityCell) -> list:
        status = [f"{facility.id} ({facility.x}, {facility.y})"] 
        
        substatuses = [f"Balance: {facility.economy.total_balance}"]
        if facility.distribution is not None:
            transport_status = [ f"{AsciiWorldStatusPrinter.status(t)} {Utils.ascii_progress_bar(t.location_pointer, t.path_len()-1, 5)}" for t in facility.distribution.fleet ]
            substatuses.append( ["Fleet:", transport_status] )
            q = facility.distribution.order_queue
            inbound_orders = [ f"{order.product_id}:{order.quantity} at ${order.unit_price} -> {order.destination.id}" for order in q]
            substatuses.append( [f"Inbound orders:", inbound_orders] )
            substatuses.append( [f"Current unit price: ${facility.distribution.economy.unit_price}"] )
            
        if facility.consumer is not None:
            in_transit_units_total = sum(sum(facility.consumer.open_orders.values(), Counter()).values())
            outbound_orders = [ f"{src} -> {AsciiWorldStatusPrinter.counter(order)}" for src, order in facility.consumer.open_orders.items()]
            substatuses.append( [f"Outbound orders ({in_transit_units_total} units):", outbound_orders] )
            substatuses.append( [f"Total units purchased: {facility.consumer.economy.total_units_purchased}"] )
            substatuses.append( [f"Total units received: {facility.consumer.economy.total_units_received}"] )
            
        if facility.seller is not None:
            substatuses.append( [f"Current unit price: ${facility.seller.economy.unit_price}"] )
            substatuses.append( [f"Current demand: {facility.seller.economy.market_demand(facility.seller.economy.unit_price)}"] )
            substatuses.append( [f"Total units sold: {facility.seller.economy.total_units_sold}"] )
            
        
        substatuses.append(["Storage:", AsciiWorldStatusPrinter.status(facility.storage) ])
        status.append(substatuses)
        return status
    
    @dispatch(ws.Transport)
    def status(t: ws.Transport) -> str:
        status = None
        if t.destination is None:
            status = "IDLE"
        else:
            if t.location_pointer == 0 and t.payload == 0:
                status = f"LOAD {t.product_id}:{t.requested_quantity} -> {t.destination.id}"
            if t.payload > 0 and t.step > 0:
                status = f"MOVE {t.product_id}:{t.payload} -> {t.destination.id}"
            if t.location_pointer == len(t.path) - 1 and t.payload > 0:
                status = f"UNLD {t.product_id}:{t.payload} -> {t.destination.id}"
            if t.step < 0 and t.payload == 0:
                status = f"BACK {t.destination.id} -> home" 
        return status
    
    @dispatch(ws.StorageUnit)
    def status(storage: ws.StorageUnit) -> list:
        return [f"Usage: {Utils.ascii_progress_bar(storage.used_capacity(), storage.max_capacity)}",
                f"Storage cost/unit: {storage.economy.unit_storage_cost}",
                f"Inventory: {AsciiWorldStatusPrinter.counter(storage.stock_levels)}"]
    
    def counter(counter) -> str:
        return dict(counter + Counter()) # this removes zero counters

    
class AsciiWorldRenderer(WorldRenderer):
    def render(self, world):
        ascii_layers = []
        
        def new_layer():
            return [[' ' for x in range(world.size_x)] for y in range(world.size_y)] 

        # print infrastructure (background)
        layer = new_layer()
        for y in range(world.size_y):
            for x in range(world.size_x):
                c = world.grid[x][y]
                if isinstance(c, ws.RailroadCell):
                    layer[y][x] = self.railroad_sprite(x, y, world.grid)
        ascii_layers.append(layer)

        # print vechicles
        layer = new_layer()
        for y in range(world.size_y):
            for x in range(world.size_x):
                c = world.grid[x][y]
                if isinstance(c, ws.FacilityCell) and c.distribution is not None:
                    for vechicle in c.distribution.fleet:
                        if vechicle.is_enroute():              
                            location = vechicle.current_location()
                            layer[location[1]][location[0]] = '*'
        ascii_layers.append(layer)
                            
        # print facilities (foreground)
        layer = new_layer()
        for y in range(world.size_y):
            for x in range(world.size_x):
                c = world.grid[x][y]
                if isinstance(c, ws.SteelFactoryCell):
                    layer[y][x] = 'S' 
                if isinstance(c, ws.LumberFactoryCell):
                    layer[y][x] = 'L' 
                if isinstance(c, ws.ToyFactoryCell):
                    layer[y][x] = 'T' 
                if isinstance(c, ws.WarehouseCell):
                    layer[y][x] = 'W' 
                if isinstance(c, ws.RetailerCell):
                    layer[y][x] = 'R' 
        ascii_layers.append(layer)

        # print ascii on canvas
        margin_side = 150 
        margin_top = 20
        font = ImageFont.truetype("resources/FiraMono-Bold.ttf", 24)
        #font = ImageFont.truetype("resources/monaco.ttf", 24)
        
        test_text = "\n".join(''.join(row) for row in ascii_layers[0])
        test_img = PIL.Image.new('RGB', (10, 10))
        test_canvas = ImageDraw.Draw(test_img)
        (map_w, map_h) = test_canvas.multiline_textsize(test_text, font=font)
        img_w = map_w + 2 * margin_side
        img_h = int(map_h * 4.0)
        
        img = PIL.Image.new('RGB', (img_w, img_h), color='#263238')
        canvas = ImageDraw.Draw(img)
        
        color_theme = [ '#80A7FB', # pale blue
                        '#FFCB6B', # yellow
                        '#C3E88D', # light green
                        '#FF5370'] # red
        for layer, color in zip(ascii_layers, color_theme):
            text = "\n".join(''.join(row) for row in layer)
            canvas.multiline_text((margin_side, margin_top), text, font=font, fill=color)
        
        # print logo
        logo = PIL.Image.open('resources/world-of-supply-logo.png', 'r').convert("RGBA")
        logo.thumbnail((img_w/5, img_h/10), PIL.Image.ANTIALIAS)
        img.paste(logo, (int(img_w/2 - img_w/10), 0), mask=logo)
        
        # print status
        font = ImageFont.truetype("resources/monaco.ttf", 11)
        status = AsciiWorldStatusPrinter.status(world)
        n_columns = 3
        n_rows = math.ceil(len(status) / n_columns)
        col_wide = img_w/n_columns * 0.90
        for i in range(n_columns):
            column_left_x = img_w/2 - (n_columns * col_wide)/2 + col_wide*i
            canvas.multiline_text((column_left_x, map_h * 1.1), 
                                  self.to_yaml(status[i*n_rows : (i+1)*n_rows]), 
                                  font=font, fill='#BBBBBB')
        
        return img
    
    def to_yaml(self, obj):
        return yaml.dump(obj).replace("'", '')

    def railroad_sprite(self, x, y, grid):
        top = False
        bottom = False
        left = False
        right = False

        if isinstance(grid[x-1][y], ws.RailroadCell):
            left = True
        if isinstance(grid[x+1][y], ws.RailroadCell):
            right = True
        if isinstance(grid[x][y-1], ws.RailroadCell):
            top = True
        if isinstance(grid[x][y+1], ws.RailroadCell):
            bottom = True

        # Sprites: ╔╗╚╝╠╣╦╩╬═║
        if (top or bottom) and not right and not left:
            return '║'
        if (right or left) and not top and not bottom:
            return '═'  
        if top and not bottom and right and not left:
            return '╚'
        if top and not bottom and not right and left:
            return '╝' 
        if bottom and not top and right and not left:
            return '╔' 
        if bottom and not top and not right and left:
            return '╗'
        if top and bottom and not right and left:
            return '╣'
        if top and bottom and right and not left:
            return '╠'
        if top and not bottom and right and left:
            return '╩'
        if bottom and not top and right and left:
            return '╦'
        if top and bottom and right and left:
            return '╬'  