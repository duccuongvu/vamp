from pathlib import Path

import numpy as np
from fire import Fire

import vamp
from viser_utils import add_spheres, add_trajectory, setup_viser_with_robot

START = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# GOAL = [0.089, 0.67, 0.0, -1.24, 0.0, 0.0, 0.0]
GOAL_DEG = [-44.5, -31.8,  37.2, -70.7,  -9.8,  0.1,  19.8]
# GOAL_DEG
# GOAL_DEG = [5.0993, 38.3882, 0, -71.0468, 0, 0, 0]
GOAL = np.deg2rad(GOAL_DEG)
OBSTACLE_CENTER = [4, 0.0, 8]
OBSTACLE_RADIUS = 0.03


def main(
    obstacle_radius: float = OBSTACLE_RADIUS,
    planner: str = "prm",
    sampler_name: str = "halton",
    **kwargs,
):
    (vamp_module, planner_func, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs("iiwa", planner, **kwargs)
    )

    robot_dir = Path(__file__).parent.parent / "resources" / "iiwa"
    server, robot = setup_viser_with_robot(robot_dir, "iiwa_spherized.urdf")
    robot.update_cfg(START)

    environment = vamp.Environment()
    environment.add_sphere(vamp.Sphere(OBSTACLE_CENTER, obstacle_radius))

    add_spheres(
        server,
        np.array([OBSTACLE_CENTER]),
        np.array([obstacle_radius]),
        colors = [[255, 0, 0]],
        prefix = "obstacle",
        )

    if not vamp.iiwa.validate(START, environment) or not vamp.iiwa.validate(GOAL, environment):
        raise RuntimeError("Start or goal configuration is in collision.")

    sampler = getattr(vamp_module, sampler_name)()
    result = planner_func(START, GOAL, environment, plan_settings, sampler)
    if not result.solved:
        raise RuntimeError("Planning failed.")

    simplified = vamp_module.simplify(result.path, environment, simp_settings, sampler)
    simplified.path.interpolate_to_resolution(vamp_module.resolution())

    stats = vamp.results_to_dict(result, simplified)
    print(
        f"""Solved in {stats['planning_iterations']} iterations
Initial path cost:   {stats['initial_path_cost']:.3f}
Simplified path cost: {stats['simplified_path_cost']:.3f}
Open http://localhost:8080 in a browser to view the trajectory."""
    )

    add_trajectory(server, simplified.path.numpy(), robot, [], [[]])

    while True:
        continue


if __name__ == "__main__":
    Fire(main)
