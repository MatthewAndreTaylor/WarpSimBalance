import warp as wp
import warp.sim.render
import numpy as np

@wp.kernel
def set_cart_kernel(body_q: wp.array(dtype=wp.transform), control_x: float, control_z: float):
    x = control_x  # Scale control signal
    z = control_z  # Scale control signal
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

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False
        self.model.joint_attach_ke = 100.0
        self.model.joint_attach_kd = 1.0

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.renderer = wp.sim.render.SimRendererOpenGL(self.model, "example4", headless=False)
        self.state = self.model.state()

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def create_cartpole(self, builder):
        """Create cartpole system using pure Python/Warp API"""
        pole_size = wp.vec3(0.04, 1.0, 0.06)
        cart_body = builder.add_body(
            origin=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
            m=0.0
        )
        builder.add_shape_sphere(
            body=cart_body,
            radius=0.1,
            density=0.0
        )
        pole_body = builder.add_body(
            origin=wp.transform(wp.vec3(0.0, 2.5, 0.0), wp.quat_identity()),
        )
        builder.add_shape_box(
            body=pole_body,
            pos=wp.vec3(0.0, 0.0, 0.0),
            hx=pole_size[0] / 2.0,
            hy=pole_size[1] / 2.0,
            hz=pole_size[2] / 2.0,
            density=100.0
        )
        builder.add_joint_ball(
            parent=cart_body,
            child=pole_body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, -0.5, 0.0), wp.quat_identity()),
        )

    def set_cart_trajectory(self, state, action):
        wp.launch(
            kernel=set_cart_kernel,
            dim=1,
            inputs=[state.body_q, action[0], action[1]],
            device=state.body_q.device
        )

    def simulate(self):
        for _ in range(self.sim_substeps):  
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)

    def step(self):
        r = 1.0  # radius of circular trajectory
        omega = 0.1  # angular frequency
        t = self.sim_time
        x = r * wp.cos(omega * t)
        z = r * wp.sin(omega * t)

        self.set_cart_trajectory(self.state, [x, z])

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state)
        self.renderer.end_frame()
    
    def get_state_vector(self):
        cart_id = 0
        pole_id = 1
        
        # convert transforms to numpy
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()

        # --- Cart ---
        cart_pos = body_q[cart_id] # [px, py, pz, qx, qy, qz, qw]
        cart_pos = cart_pos[[0, 2]]

        # --- Pole ---
        pole_quat = body_q[pole_id][3:]  # quaternion part
        pole_vel = body_qd[pole_id][3:]  # angular velocity part

        return np.concatenate([cart_pos, pole_quat, pole_vel])
        
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-frames", type=int, default=1000, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        for _ in range(args.num_frames):
            example.step()
            example.render()

        example.renderer.save()
