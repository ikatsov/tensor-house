import numpy as np
from tqdm import tqdm as tqdm
import importlib

import world_of_supply_environment as ws
importlib.reload(ws)
import world_of_supply_renderer as wsr
importlib.reload(wsr)

if __name__ == "__main__":
    # Measure the simulation rate, steps/sec
    world = ws.WorldBuilder.create(80, 16)
    policy = ws.SimpleControlPolicy()
    for i in tqdm(range(10000)):
        world.act(policy.compute_control(world))

    # Test rendering
    renderer = wsr.AsciiWorldRenderer()
    frame_seq = []
    world = ws.WorldBuilder.create(80, 16)
    policy = ws.SimpleControlPolicy()
    for epoch in tqdm(range(300)):
        frame = renderer.render(world)
        frame_seq.append(np.asarray(frame))
        world.act(policy.compute_control(world))

    print('Rendering the animation...')
    wsr.AsciiWorldRenderer.plot_sequence_images(frame_seq)