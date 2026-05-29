import numpy as np
from pathlib import Path

from fire import Fire

import vamp
from vamp import pybullet_interface as vpb


START = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# GOAL = [0.089, 0.67, 0.0, -1.24, 0.0, 0.0, 0.0]
GOAL_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 119.0, 0.0]
GOAL = np.deg2rad(GOAL_DEG)
OBSTACLE_CENTER = [0.4, 0.0, 0.8]
OBSTACLE_RADIUS = 0.03


def main(
    planner: str = "rrtc",
    sampler_name: str = "halton",
    visualize: bool = False,
    **kwargs,
):
    (vamp_module, planner_func, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs("iiwa", planner, **kwargs)
    )

    sampler = getattr(vamp_module, sampler_name)()
    environment = vamp.Environment()
    environment.add_sphere(vamp.Sphere(OBSTACLE_CENTER, OBSTACLE_RADIUS))

    start_valid = vamp.iiwa.validate(START, environment)
    goal_valid = vamp.iiwa.validate(GOAL, environment)
    print(f"Start valid: {start_valid}")
    print(f"Goal valid:  {goal_valid}")

    if not start_valid or not goal_valid:
        raise RuntimeError("Start or goal configuration is in collision.")

    result = planner_func(START, GOAL, environment, plan_settings, sampler)
    print(f"Solved: {result.solved}")
    print(f"Planning iterations: {result.iterations}")
    print(f"Initial path waypoints: {len(result.path)}")

    if not result.solved:
        return

    simplified = vamp_module.simplify(result.path, environment, simp_settings, sampler)
    stats = vamp.results_to_dict(result, simplified)
    print(
        f"""
Planning Time: {stats['planning_time'].microseconds:8d}μs
Simplify Time: {stats['simplification_time'].microseconds:8d}μs
   Total Time: {stats['total_time'].microseconds:8d}μs

Planning Iters: {stats['planning_iterations']}

Path Length:
   Initial: {stats['initial_path_cost']:5.3f}
Simplified: {stats['simplified_path_cost']:5.3f}
"""
    )

    if visualize:
        robot_dir = Path(__file__).parent.parent / "resources" / "iiwa"
        sim = vpb.PyBulletSimulator(
            str(robot_dir / "iiwa_spherized.urdf"),
            vamp_module.joint_names(),
            True,
        )
        sim.add_sphere(OBSTACLE_RADIUS, OBSTACLE_CENTER)

        path = simplified.path
        path.interpolate_to_resolution(vamp_module.resolution())
        sim.animate(path)


if __name__ == "__main__":
    Fire(main)
