import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

class QuinticPlanner:
    """
    【规划层】大脑：基于有限状态机 (FSM) 和五次多项式的工业 Pick & Place 规划器
    """
    def __init__(self):
        self.state = "INIT"
        self.state_start_time = 0.0
        
        # ==========================================
        # 坐标高度对齐
        # 方块中心在 Z=0.025，直接让 TCP 对准中心偏上一点点即可
        # ==========================================
        self.p_home = np.array([0.3, 0.0, 0.5])
        self.p_above_pick = np.array([0.5, 0.0, 0.2])
        self.p_pick = np.array([0.5, 0.0, 0.028])  # 完美抓取高度 (留3毫米容差防撞地)
        self.p_above_place = np.array([0.5, 0.3, 0.2])
        self.p_place = np.array([0.5, 0.3, 0.03])  # 放置高度
        
        self.p_start = self.p_home.copy()
        self.p_end = self.p_home.copy()
        self.duration = 1.0
        
        # ==========================================
        # 夹爪初始信号
        # 255.0 是张开，0.0 是闭合。初始状态保持张开
        # ==========================================
        self.gripper_cmd = 255.0 

    def _quintic_spline(self, t, T, p0, p1):
        """生成五次多项式插值曲线"""
        s = np.clip(t / T, 0.0, 1.0)
        c0, c1, c2 = 10.0, -15.0, 6.0
        
        pos = p0 + (p1 - p0) * (c0*s**3 + c1*s**4 + c2*s**5)
        vel = (p1 - p0) * (30*s**2 - 60*s**3 + 30*s**4) / T
        acc = (p1 - p0) * (60*s - 180*s**2 + 120*s**3) / (T**2)
        
        return pos, vel, acc

    def transition(self, next_state, target_pos, duration, current_time):
        self.state = next_state
        self.p_start = self.p_end.copy()
        self.p_end = target_pos
        self.duration = duration
        self.state_start_time = current_time
        print(f"[FSM] 切换至状态: {self.state}")

    def get_trajectory(self, t):
        elapsed = t - self.state_start_time
        
        # 状态机逻辑
        if self.state == "INIT" and t > 1.0:
            self.transition("APPROACH", self.p_above_pick, 2.0, t)
            
        elif self.state == "APPROACH" and elapsed > self.duration + 0.5:
            self.transition("DESCEND", self.p_pick, 1.5, t)
            
        elif self.state == "DESCEND" and elapsed > self.duration + 0.5:
            self.gripper_cmd = 0.0  # 修正：下发 0.0 闭合夹爪，咬住方块
            self.transition("GRASP_WAIT", self.p_pick, 1.0, t)
            
        elif self.state == "GRASP_WAIT" and elapsed > self.duration:
            self.transition("LIFT", self.p_above_pick, 1.5, t)
            
        elif self.state == "LIFT" and elapsed > self.duration + 0.2:
            self.transition("MOVE", self.p_above_place, 2.0, t)
            
        elif self.state == "MOVE" and elapsed > self.duration + 0.5:
            self.transition("PLACE_DESCEND", self.p_place, 1.5, t)
            
        elif self.state == "PLACE_DESCEND" and elapsed > self.duration + 0.5:
            self.gripper_cmd = 255.0  # 修正：下发 255.0 张开夹爪，释放方块
            self.transition("RELEASE_WAIT", self.p_place, 1.0, t)
            
        elif self.state == "RELEASE_WAIT" and elapsed > self.duration:
            self.transition("RETURN", self.p_home, 2.0, t)

        return *self._quintic_spline(elapsed, self.duration, self.p_start, self.p_end), self.gripper_cmd


class IndustrialController:
    """
    【控制层】脊髓：导纳滤波器 + 闭环微分IK + 计算力矩控制(CTC)
    """
    def __init__(self, model, data, dt, ee_name="hand"):
        self.model = model
        self.data = data
        self.dt = dt
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        
        self.jacp = np.zeros((3, self.model.nv))
        self.jacp_dot = np.zeros((3, self.model.nv))
        self.M = np.zeros((self.model.nv, self.model.nv))
        self.q_d = None
        
        # 导纳参数 (Admittance)
        self.M_d_inv = np.linalg.inv(np.diag([1.0, 1.0, 1.0]))
        self.K_d = np.diag([800.0, 800.0, 800.0]) 
        self.B_d = np.diag([56.5, 56.5, 56.5]) 
        self.x_c = np.zeros(3)
        self.dx_c = np.zeros(3)
        
    def compute_torque(self, x_tar_orig, dx_tar_orig, ddx_tar_orig, Kp_joint, Kd_joint, F_ext) -> tuple:
        # ==========================================
        # 0. 导纳滤波器 (防爆约束)
        # ==========================================
        F_ext_clipped = np.clip(F_ext, -40.0, 40.0) # 限制异常脉冲力
        ddx_c = self.M_d_inv @ (F_ext_clipped - self.B_d @ self.dx_c - self.K_d @ self.x_c)
        
        self.dx_c += ddx_c * self.dt
        self.x_c += self.dx_c * self.dt
        self.x_c = np.clip(self.x_c, -0.1, 0.1) # 限制最大退让位移为 10cm
        
        x_tar = x_tar_orig + self.x_c
        dx_tar = dx_tar_orig + self.dx_c
        ddx_tar = ddx_tar_orig + ddx_c

        # ==========================================
        # 1. 运动学映射与闭环 IK
        # ==========================================
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        hand_pos = self.data.xpos[self.ee_id]
        hand_mat = self.data.xmat[self.ee_id].reshape(3, 3)
        tcp_offset = np.array([0.0, 0.0, 0.1034])
        x_curr = hand_pos + hand_mat @ tcp_offset
        
        mujoco.mj_jac(self.model, self.data, self.jacp, None, x_curr, self.ee_id)
        J_p = self.jacp[:, :7]
        
        q_curr = self.data.qpos[:7]
        dq_curr = self.data.qvel[:7]
        
        if self.q_d is None:
            self.q_d = q_curr.copy()

        J_pinv = np.linalg.pinv(J_p)
        err_x = x_tar - x_curr
        
        # 速度积分与零空间自运动
        dq_d_primary = J_pinv @ (dx_tar + 15.0 * err_x)
        N_T = np.eye(7) - J_pinv @ J_p
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        dq_d_null = N_T @ (5.0 * (q_home - q_curr))
        
        dq_d = dq_d_primary + dq_d_null
        self.q_d += dq_d * self.dt
        
        # 目标加速度
        mujoco.mj_jacDot(self.model, self.data, self.jacp_dot, None, x_curr, self.ee_id)
        J_dot_q_dot = self.jacp_dot[:, :7] @ dq_curr
        ddq_d = J_pinv @ (ddx_tar - J_dot_q_dot)

        # ==========================================
        # 2. 完美的计算力矩控制 (CTC)
        # ==========================================
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        M_7 = self.M[:7, :7]
        
        e_q = self.q_d - q_curr
        de_q = dq_d - dq_curr
        
        # 将前馈加速度与 PD 纠偏加速度统一合并后，乘以惯性矩阵
        ddq_cmd = ddq_d + Kp_joint * e_q + Kd_joint * de_q
        tau_final = M_7 @ ddq_cmd + self.data.qfrc_bias[:7]
        
        return tau_final, np.linalg.norm(err_x)
    
    def get_tcp_external_force(self):
        # 1. 获取法兰body的外力 + 外力矩 (空间力 wrench)
        # data.cfrc_ext: [fx,fy,fz,tx,ty,tz] 作用在body坐标系原点(hand法兰)
        wrench_hand = self.data.cfrc_ext[self.ee_id]
        f_hand = wrench_hand[:3]
        tau_hand = wrench_hand[3:]

        # TCP在法兰局部坐标系偏移：z向0.1034m
        tcp_local = np.array([0, 0, 0.1034])
        hand_rot = self.data.xmat[self.ee_id].reshape(3,3)
        tcp_world = hand_rot @ tcp_local

        # 力不随平移改变；力矩需要平移耦合项 r × f
        f_tcp = f_hand.copy()
        tau_tcp = tau_hand + np.cross(tcp_world, f_hand)

        # 只返回三维力（你导纳只用平移力）
        return f_tcp


class FrankaSimNode:
    def __init__(self, xml_path="scene.xml"):
        # 当前脚本所在文件夹
        cur_script_dir = Path(__file__).resolve().parent
        # 拼接xml完整路径
        full_xml_path = cur_script_dir / xml_path

        if not full_xml_path.exists():
            raise FileNotFoundError(f"XML 文件不存在: {full_xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(full_xml_path))
        self.data = mujoco.MjData(self.model)
        
        self.dt = self.model.opt.timestep
        self.controller = IndustrialController(self.model, self.data, self.dt)
        self.planner = QuinticPlanner()
        
        # 极高的底层刚度，保障 CTC 追踪精度
        self.Kp_joint = np.array([2500.0, 2500.0, 2500.0, 2500.0, 225.0, 225.0, 225.0])
        self.Kd_joint = np.array([100.0, 100.0, 100.0, 100.0, 30.0, 30.0, 30.0])
        
        self._reset_home_pose()
        
    def _reset_home_pose(self):
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = q_home
        mujoco.mj_forward(self.model, self.data)
        
    def run(self):
        print("🚀 启动自动化搬运任务...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            real_start_time = time.time()
            
            while viewer.is_running():
                real_elapsed = time.time() - real_start_time
                
                while self.data.time < real_elapsed:
                    t = self.data.time
                    
                    # 1. 大脑：生成轨迹和夹爪控制指令
                    x_tar, dx_tar, ddx_tar, grip_cmd = self.planner.get_trajectory(t)
                    
                    # 2. 传感器：估算外部受力 (工程简化版：不带力传感器时用零代替，靠高阻抗硬抗)
                    # 如果有六轴力传感器，这里应传入真实的 F_ext
                    F_ext = np.zeros(3) 
                    F_ext = self.controller.get_tcp_external_force()  # 获取 TCP 外力
                    
                    # 3. 脊髓：计算底层关节力矩
                    tau, error = self.controller.compute_torque(
                        x_tar, dx_tar, ddx_tar, self.Kp_joint, self.Kd_joint, F_ext
                    )
                    
                    # 4. 肌肉执行
                    self.data.ctrl[:7] = tau
                    self.data.ctrl[7] = grip_cmd #[cite: 3]
                    mujoco.mj_step(self.model, self.data)
                    
                viewer.sync()

if __name__ == "__main__":
    sim_node = FrankaSimNode()
    sim_node.run()