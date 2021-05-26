#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini, Matteo Tiezzi
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# SAILenv is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages

import random
import cv2

# Import src
from sailenv import Vector3
from sailenv.agent import Agent
from sailenv.dynamics.uniform_movement_random_bounce import UniformMovementRandomBounce
from sailenv.dynamics.waypoints import CatmullWaypoints, Waypoint, LinearWaypoints
from sailenv.generators.object import Object
from sailenv.generators.scenario import Scenario
from sailenv.generators.timings import AllTogetherTimings, WaitUntilCompleteTimings, DictTimings

host = "127.0.0.1"
if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False, width=256, height=192, host=host, port=8085, use_gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    last_unity_time: float = 0.0

    try:
        print(f"Available scenes: {agent.scenes}")
        scene = "solid_benchmark/scene"
        #print(f"Changing scene to {scene}")
        #agent.change_scene(scene)

        #print(agent.spawnable_objects_names)

        # agent.spawn_collidable_view_frustum()

        agent_pos = agent.get_position()

        object_pos = agent_pos + (0, 0, 1)

        waypoints = []
        for i in range(4):
            for j in range(3):
                waypoints.append(
                    Waypoint(
                        Vector3(5 - i, 3 - j, 4 - i + j),
                        Vector3(2 + i, 5 - 2 * j, 2 + i - j)
                    )
                )

        dynamic1 = CatmullWaypoints(waypoints=waypoints)
        dynamic2 = LinearWaypoints(waypoints=waypoints)
        dynamic3 = UniformMovementRandomBounce(seed=32, start_direction=Vector3(0, 5, 2))

        objects = [
            Object("cylinder", "Cylinder", Vector3(0, 0, 0), Vector3(0, 0, 0), dynamic1, frustum_limited=False),
            Object("cylinder2", "Cylinder", Vector3(0, 0, 2), Vector3(0, 0, 0), dynamic2, frustum_limited=False),
            # Object("sphere", "Cylinder", Vector3(0, 0, 2), Vector3(0, 0, 0), dynamic3),
        ]

        timings1 = AllTogetherTimings(0.75)
        timings2 = WaitUntilCompleteTimings(1.5)
        timings3 = DictTimings({
            0: 10.,
            1: 5.,
            2: 7.5
        })

        scenario = Scenario(scene, objects, timings2, True)

        spawned_ids = agent.load_scenario(scenario)

        print(spawned_ids)

        input()

        agent.despawn_object(spawned_ids[1][0])

    finally:
        print(f"Closing agent {agent.id}")
        agent.delete()
