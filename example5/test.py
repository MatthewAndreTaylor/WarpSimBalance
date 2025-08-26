import warp as wp
import warp.sim.render
import numpy as np
import enum


@wp.kernel
def set_cart_kernel(
    body_q: wp.array(dtype=wp.transform), control_x: float, control_z: float
):
    x = control_x  # Scale control signal
    z = control_z  # Scale control signal
    y = 2.0
    body_q[0] = wp.transform(wp.vec3(x, y, z), wp.quat_identity())


class ActionSpaceType(enum.Enum):
    CONTINUOUS = enum.auto()
    DISCRETE = enum.auto()


def make_actions(N):
    """N evenly spaced unit directions on the 2D unit circle, returned as wp.vec2."""
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    actions = [wp.vec2(np.cos(theta), np.sin(theta)) for theta in thetas]
    actions.append(wp.vec2(0.0, 0.0))

    multiplier = 0.01
    actions = [action * multiplier for action in actions]

    return actions


class Example:
    def __init__(self):
        builder = wp.sim.ModelBuilder(gravity=-3.0)
        self.create_cartpole(builder)

        self.action_space_type = ActionSpaceType.DISCRETE
        # self.actions = make_actions(16)
        self.actions = [
            wp.vec2(0.0, 0.0),  # No movement
            wp.vec2(0.01, 0.0),  # Move right
            wp.vec2(-0.01, 0.0),  # Move left
            wp.vec2(0.0, 0.01),  # Move up
            wp.vec2(0.0, -0.01),  # Move down
            wp.vec2(0.01, 0.01),  # Move right-up
            wp.vec2(-0.01, 0.01),  # Move left-up
            wp.vec2(0.01, -0.01),  # Move right-down
            wp.vec2(-0.01, -0.01),  # Move left-down
        ]

        self.sim_time = 0.0
        self.current_x = 0.0
        self.current_z = 0.0

        fps = 120
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 120
        self.sim_dt = self.frame_dt / self.sim_substeps

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False
        self.model.joint_attach_ke = 150.0
        self.model.joint_attach_kd = 1.0

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.renderer = wp.sim.render.SimRendererOpenGL(
            self.model, "example", headless=False
        )
        self.state = self.model.state()

        self.use_cuda_graph = wp.get_device().is_cuda and False
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self.step(action=1)

    def create_cartpole(self, builder):
        """Create cartpole system using pure Python/Warp API"""
        pole_size = wp.vec3(0.04, 1.0, 0.06)
        cart_body = builder.add_body(
            origin=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()), m=0.0
        )
        builder.add_shape_sphere(body=cart_body, radius=0.1, density=0.0)
        pole_body = builder.add_body(
            origin=wp.transform(wp.vec3(0.0, 2.5, 0.0), wp.quat_identity()),
        )
        builder.add_shape_box(
            body=pole_body,
            pos=wp.vec3(0.0, 0.0, 0.0),
            hx=pole_size[0] / 2.0,
            hy=pole_size[1] / 2.0,
            hz=pole_size[2] / 2.0,
            density=50.0,
        )
        builder.add_joint_ball(
            parent=cart_body,
            child=pole_body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, -0.5, 0.0), wp.quat_identity()),
        )

    def set_cart_trajectory(self, state, action):

        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            self.current_x += action[0]
            self.current_z += action[1]

        elif self.action_space_type == ActionSpaceType.DISCRETE:
            discrete_action = self.actions[action]
            self.current_x += discrete_action[0]
            self.current_z += discrete_action[1]

        wp.launch(
            kernel=set_cart_kernel,
            dim=1,
            inputs=[state.body_q, self.current_x, self.current_z],
            device=state.body_q.device,
        )

    def reset(self):
        self.current_x = 0.0
        self.current_z = 0.0
        self.sim_time = 0.0
        builder = wp.sim.ModelBuilder(gravity=-3.0)
        self.create_cartpole(builder)
        self.model = builder.finalize()
        self.model.joint_attach_ke = 150.0
        self.model.joint_attach_kd = 1.0
        self.state = self.model.state()
        self.step(action=1)
        return self.get_state_vector()

    def step(self, action):
        """Apply action and step the simulation for one environment timestep.

        Returns (observation, reward, terminated)
        """
        self.set_cart_trajectory(self.state, action)

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        obs = self.get_state_vector()

        pole_quat = obs[2:6]  # quaternion part (qx, qy, qz, qw)

        # Check if the pole has fallen by computing the tilt angle from the
        # quaternion. For the upright pole the quaternion is identity (angle=0).
        # We compute angle = 2*arccos(qw). If it exceeds a threshold the pole
        # is considered fallen.
        qw = float(np.clip(pole_quat[3], -1.0, 1.0))
        tilt = 2.0 * np.arccos(qw)  # radians, in [0, pi]

        # threshold: 75 degrees (tunable)
        max_tilt = np.deg2rad(75.0)
        terminated = bool(tilt > max_tilt)
        reward = 1.0

        return obs, reward, terminated

    def render(self):
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state)
        self.renderer.end_frame()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(
                self.model, self.state, self.state, self.sim_dt
            )

    def get_state_vector(self):
        cart_id = 0
        pole_id = 1

        # convert transforms to numpy
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()

        # --- Cart ---
        cart_pos = body_q[cart_id]  # [px, py, pz, qx, qy, qz, qw]
        cart_pos = cart_pos[[0, 2]]

        # --- Pole ---
        pole_quat = body_q[pole_id][3:]  # quaternion part
        pole_vel = body_qd[pole_id][3:]  # angular velocity part

        return np.concatenate([cart_pos, pole_quat, pole_vel])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument(
        "--num-frames", type=int, default=1500, help="Total number of frames."
    )
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        terminated = False

        for i in range(args.num_frames):
            if not terminated:

                if i < 200:
                    action = 0
                elif i < 255:  # 260
                    action = 2
                else:
                    action = 0

                obs, reward, terminated = example.step(action=action)
                print(f"Step {i}")

            example.render()

            # print("State Vector:", obs)
            # print("Reward:", reward)
            # print("Terminated:", terminated)

        state = example.reset()
        print("Reset State Vector:", state)
        example.render()

        example.renderer.save()
