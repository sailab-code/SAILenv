from sailenv import Vector3
from sailenv.dynamics.uniform_movement_random_bounce import UniformMovementRandomBounce
from sailenv.dynamics.waypoints import Waypoint, CatmullWaypoints, LinearWaypoints
from sailenv.generators.object import Object
from sailenv.generators.scenario import Scenario, Frustum
from sailenv.generators.timings import AllTogetherTimings, WaitUntilCompleteTimings, DictTimings


def flying_cylinder_empty(agent_pos):
    waypoints_offsets = [
        Waypoint(Vector3(0., 0., 4.), Vector3(0., 0., 0.)),
        Waypoint(Vector3(0., 2., 4.), Vector3(90., 0., 0.)),
        Waypoint(Vector3(3., 2., -4.), Vector3(90., 90., 0.)),
        Waypoint(Vector3(3., 1., -7.), Vector3(90., 90., 90.)),
        Waypoint(Vector3(-5., 1., 7.), Vector3(90., 90., 180.)),
    ]

    waypoints = [Waypoint(offset.position + agent_pos, offset.rotation) for offset in waypoints_offsets]

    dynamic1 = CatmullWaypoints(waypoints=waypoints, total_time=10.)

    objects = [
        Object("cylinder", "Cylinder", Vector3(0, 0, 2), Vector3(0, 0, 0), dynamic1),
    ]

    return Scenario("object_view/scene", objects)


def room_01():
    waypoints = [
        Waypoint(Vector3(0.5, 1.4, 0.5), Vector3(0., 0., 0.)),
        Waypoint(Vector3(0.3, 1., -1.), Vector3(90., 0., 0.)),
        Waypoint(Vector3(-0.4, 0.6, 0.5), Vector3(90., 90., 0.)),
        Waypoint(Vector3(-1.5, 1.8, -1.0), Vector3(90., 90., 90.)),
        Waypoint(Vector3(0.5, 1.4, -1.0), Vector3(90., 90., 180.))
    ]

    dynamic = CatmullWaypoints(waypoints=waypoints)

    objects = [
        Object("racket", "Tennis Racket 01", Vector3(0.5, 1.4, 0.5), Vector3(0., 0., 0.), dynamic)
    ]

    timings = WaitUntilCompleteTimings(1.5)

    scenario = Scenario("room_01/scene", objects, timings, Frustum(False, 20.))
    return scenario


def all_together(agent_pos):
    dynamic1 = UniformMovementRandomBounce(seed=32, speed=0.5, start_direction=Vector3(0, 5, 2))
    dynamic2 = UniformMovementRandomBounce(seed=32, speed=0.5, start_direction=Vector3(2, 3, 1))
    dynamic3 = UniformMovementRandomBounce(seed=32, speed=0.5, start_direction=Vector3(4, 1, 2))

    objects = [
        Object("chair", "Chair 01", agent_pos + Vector3(0.5, 0., 2.), Vector3(0., 0., 0.), dynamic1, frustum_limited=True),
        Object("pillow", "Pillow 01", agent_pos + Vector3(0., -0.5, 2.), Vector3(0., 0., 0.), dynamic2, frustum_limited=True),
        Object("dish", "Dish 01", agent_pos + Vector3(-0.5, 0.5, 2.), Vector3(0., 0., 0.), dynamic3, frustum_limited=True),
    ]

    timings = AllTogetherTimings(0.75)
    return Scenario("room_02/scene", objects, timings, Frustum(True, 2.5))
