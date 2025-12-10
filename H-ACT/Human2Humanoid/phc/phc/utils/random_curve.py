import torch
import random
from scipy.spatial.transform import Rotation as R
import inspect

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)


def bezier_interp(p0, p1, p2, p3, t):
    """Cubic Bezier interpolation at time t"""
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + \
           3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def interpolate_bezier(array_t: torch.Tensor, keyframes: list):
    """Interpolates keyframes using Bezier curves over the length of array_t"""
    t_len, n_dim = array_t.shape
    out = torch.zeros((t_len, n_dim), dtype=torch.float32, device=array_t.device)

    # Sort and extract keyframe data
    keyframes = sorted(keyframes, key=lambda x: x[0])
    ratios = [kf[0] for kf in keyframes]
    values = [torch.tensor(kf[1], dtype=torch.float32, device=array_t.device) for kf in keyframes]

    frame_ratios = torch.linspace(0, 1, t_len, device=array_t.device)

    for i in range(len(keyframes) - 1):
        r0, r1 = ratios[i], ratios[i + 1]
        v0, v1 = values[i], values[i + 1]

        # Define control points between v0 and v1
        p1 = v0 + 0.25 * (v1 - v0)
        p2 = v0 + 0.75 * (v1 - v0)

        # Determine the frame indices that fall in this interval
        mask = (frame_ratios >= r0) & (frame_ratios <= r1)
        local_ts = (frame_ratios[mask] - r0) / (r1 - r0)
        indices = torch.where(mask)[0]

        for idx, t_val in zip(indices, local_ts):
            out[idx] = bezier_interp(v0, p1, p2, v1, t_val)

    return out

class RootStateConfig(BaseConfig):
    class PosConfig:
        vel_x_mean = 0
        vel_x_std = 0.5
        vel_y_mean = 0
        vel_y_std = 0.2

        key_frame_num_range = [2, 5]

    class AngConfig:
        ang_type = "rot"
        ang_vel_yaw_mean = 0
        ang_vel_yaw_std = 0.2

        key_frame_num_range = [2, 5]

class RootStateGenerator:
    def __init__(self, config: RootStateConfig):
        self.cfg = config

    def __call__(self, t: int, device='cpu', dt=0.02):
        return self.generate_random_root_state(t, device, dt)

    def generate_random_root_state(self, t: int, device='cpu', dt=0.02):
        """
        Generates a random root state trajectory over t frames.
        Returns:
            - vel_xy: (T, 3) linear velocities (Z=0)
            - pos_xy: (T, 3) positions via cumulative integration
            - ang: (T, 4) quaternion orientation
            - ang_vel: (T, 3) angular velocities (XYZ)
        """
        vel_xy = self.generate_vel(t, device)
        pos_xy = torch.cumsum(vel_xy, dim=0) * dt
        pos_xy[:, 2] = 0.8

        ang_vel = self.generate_ang(t, device)
        angles = torch.cumsum(ang_vel, dim=0) * dt

        # Convert cumulative angles to quaternions
        angles_np = angles.cpu().numpy()
        if self.cfg.AngConfig.ang_type == "mat":
            ang_np = R.from_euler('xyz', angles_np).as_matrix()
        elif self.cfg.AngConfig.ang_type == "quat":
            ang_np = R.from_euler('xyz', angles_np).as_quat()
        else:
            ang_np = R.from_euler('xyz', angles_np).as_rotvec()
        ang = torch.tensor(ang_np, dtype=torch.float32, device=device)

        return pos_xy, ang, vel_xy, ang_vel
    
    def generate_vel(self, t: int, device='cpu'):
        """Generates a velocity curve using random Bezier keyframes"""
        num_kf = random.randint(self.cfg.PosConfig.key_frame_num_range[0], self.cfg.PosConfig.key_frame_num_range[1])
        ratios = sorted(torch.rand(num_kf).tolist())
        ratios[0], ratios[-1] = 0.0, 1.0  # ensure start and end

        kfs = []
        for r in ratios:
            vel_x = random.gauss(self.cfg.PosConfig.vel_x_mean, self.cfg.PosConfig.vel_x_std)
            vel_y = random.gauss(self.cfg.PosConfig.vel_y_mean, self.cfg.PosConfig.vel_y_std)
            kfs.append((r, [vel_x, vel_y, 0.0]))

        base = torch.zeros((t, 3), device=device)
        return interpolate_bezier(base, kfs)


    def generate_ang(self, t: int, device='cpu'):
        """Generates angular velocity (yaw-only) using random Bezier keyframes"""
        num_kf = random.randint(self.cfg.AngConfig.key_frame_num_range[0], self.cfg.AngConfig.key_frame_num_range[1])
        ratios = sorted(torch.rand(num_kf).tolist())
        ratios[0], ratios[-1] = 0.0, 1.0

        kfs = []
        for r in ratios:
            yaw = random.gauss(self.cfg.AngConfig.ang_vel_yaw_mean, self.cfg.AngConfig.ang_vel_yaw_std)
            kfs.append((r, [0.0, 0.0, yaw]))

        base = torch.zeros((t, 3), device=device)
        return interpolate_bezier(base, kfs)

# def plot_random_trajectory_with_arrows(t=120, device='cpu', arrow_interval=10):
#     cfg = RootStateConfig()
#     ge = RootStateGenerator(cfg)
#     pos_xy, ang_quat, vel_xy, ang_vel = ge.generate_random_root_state(t, device)

#     x = pos_xy[:, 0].cpu().numpy()
#     y = pos_xy[:, 1].cpu().numpy()

#     # 提取 yaw 方向（从 quaternion）
#     rot = R.from_quat(ang_quat.cpu().numpy())
#     euler = rot.as_euler('xyz', degrees=False)
#     yaw = euler[:, 2]  # Z-axis rotation (yaw)

#     plt.figure(figsize=(6, 6))
#     plt.plot(x, y, marker='o', markersize=2, linewidth=1.5, label='Trajectory')
#     plt.scatter(x[0], y[0], color='green', label='Start', zorder=5)
#     plt.scatter(x[-1], y[-1], color='red', label='End', zorder=5)

#     # 添加方向箭头（每隔 arrow_interval 帧）
#     for i in range(0, t, arrow_interval):
#         dx = np.cos(yaw[i]) * 0.2
#         dy = np.sin(yaw[i]) * 0.2
#         plt.arrow(x[i], y[i], dx, dy, head_width=0.05, head_length=0.1, fc='blue', ec='blue')

#     plt.title("Random Root Trajectory with Yaw Arrows")
#     plt.xlabel("X Position")
#     plt.ylabel("Y Position")
#     plt.axis('equal')
#     plt.grid(True)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# # 执行绘图
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# plot_random_trajectory_with_arrows(t=1200, device=device, arrow_interval=10)

cfg = RootStateConfig()
root_state_generator = RootStateGenerator(cfg)