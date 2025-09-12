import warp as wp
import warp.sim.render
import random
import math
import matplotlib.pyplot as plt

# Modified cart pole problem using warp.sim API (3 degrees of freedom)

times = []
cart_positions = []
pole_angles = []
pole_angular_velocities = []


@wp.kernel
def set_cart_kernel(body_q: wp.array(dtype=wp.transform), t: float):
    r = 2.0
    omega = 5.0  # frequency of cart oscillation (speed)
    # omega = 0.1

    x = r * wp.cos(omega * t)
    z = r * wp.sin(omega * t)
    y = 2.0
    body_q[0] = wp.transform(wp.vec3(x, y, z), wp.quat_identity())


class Example:
    def __init__(self):

        builder = wp.sim.ModelBuilder()
        self.create_cartpole(builder)

        self.sim_time = 0.0
        fps = 120
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 120
        self.sim_dt = self.frame_dt / self.sim_substeps

        # joint initial positions (random pole orientation)
        builder.joint_q[-3:] = [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
        ]

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.model.joint_attach_ke = 100.0
        self.model.joint_attach_kd = 1.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.renderer = wp.sim.render.SimRendererOpenGL(
            self.model, "example4", headless=False
        )
        # self.renderer = wp.sim.render.SimRenderer(self.model, path="example4.usd")
        self.state = self.model.state()

        wp.sim.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, None, self.state
        )

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def create_cartpole(self, builder):
        """Create cartpole system using pure Python/Warp API"""

        # Material properties
        density = 100.0  # kg/m^3

        # Pole properties (0.04 x 1.0 x 0.06 box)
        pole_size = wp.vec3(0.04, 1.0, 0.06)

        # Create cart body (kinematic - not affected by gravity)
        cart_body = builder.add_body(
            origin=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()), m=0.0
        )

        # Add cart shape
        builder.add_shape_sphere(body=cart_body, radius=0.1, density=0.0)

        # Create pole body
        pole_body = builder.add_body(
            origin=wp.transform(wp.vec3(0.0, 2.5, 0.0), wp.quat_identity()),
        )

        # Add pole shape
        builder.add_shape_box(
            body=pole_body,
            pos=wp.vec3(0.0, 0.0, 0.0),
            hx=pole_size[0] / 2.0,
            hy=pole_size[1] / 2.0,
            hz=pole_size[2] / 2.0,
            density=density,
        )

        # Create spherical joint connecting cart to pole
        builder.add_joint_ball(
            parent=cart_body,
            child=pole_body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, -0.5, 0.0), wp.quat_identity()),
        )

    def set_cart_trajectory(self, state, t):
        wp.launch(
            kernel=set_cart_kernel,
            dim=1,
            inputs=[state.body_q, t],
            device=state.body_q.device,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(
                self.model, self.state, self.state, self.sim_dt
            )

    def step(self):

        self.cart_physical_properties()

        self.set_cart_trajectory(self.state, self.sim_time)

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state)
        self.renderer.end_frame()

    def cart_physical_properties(self):
        cart_id = 0
        pole_id = 1

        # convert transforms to numpy
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()

        t = self.sim_time

        # --- Cart ---
        cart_pos = body_q[cart_id]  # [px, py, pz, qx, qy, qz, qw]

        # --- Pole ---
        pole_quat = body_q[pole_id][3:]  # quaternion part

        local_down = wp.vec3(0.0, -1.0, 0.0)
        world_down = wp.quat_rotate(pole_quat, local_down)
        global_down = wp.vec3(0.0, -1.0, 0.0)

        cos_theta = wp.dot(world_down, global_down) / (
            wp.length(world_down) * wp.length(global_down)
        )
        theta = float(math.acos(cos_theta))
        theta_dot = body_qd[pole_id][
            3:
        ]  # angular velocity around the pole's local y-axis

        times.append(t)
        cart_positions.append(cart_pos[:3])
        pole_angles.append(theta)
        pole_angular_velocities.append(theta_dot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument(
        "--num-frames", type=int, default=1000, help="Total number of frames."
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        for _ in range(args.num_frames):
            example.step()
            example.render()

        # graph the statistics
        plt.figure(figsize=(12, 8))
        plt.suptitle("Cart-Pole Simulation Statistics")
        plt.subplot(2, 2, 1)
        plt.plot(times, cart_positions, label="Cart Position")
        plt.title("Cart Position")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(times, pole_angles, label="Pole Angle")
        plt.title("Pole Angle")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (radians)")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(times, pole_angular_velocities, label="Pole Angular Velocity")
        plt.title("Pole Angular Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (radians/s)")
        plt.legend()

        plt.tight_layout()
        plt.show()

        example.renderer.save()
