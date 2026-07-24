import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from pathlib import Path

class TrajectoryGenerator:
    """
    【规划层】导纳控制：8字形打磨轨迹
    """
    @staticmethod
    def get_trajectory(t: float) -> tuple:
        A_x, omega_x = 0.04, 0.8 * np.pi
        A_y, omega_y = 0.06, 1.6 * np.pi
        
        x0 = 0.5 - A_x
        y0 = 0.0 - A_y
        z0 = 0.5
        
        x = np.array([x0, y0, z0])
        dx = np.zeros(3)
        ddx = np.zeros(3)
        
        if t < 0.5:
            pass
            
        elif t < 2.0:
            phase_t = t - 0.5
            x[2] = 0.28 + 0.22 * np.exp(-3.0 * phase_t)
            dx[2] = -0.22 * 3.0 * np.exp(-3.0 * phase_t)
            ddx[2] = 0.22 * 3.0**2 * np.exp(-3.0 * phase_t)
            
        else:
            phase_t = t - 2.0
            x[2] = 0.28 # 规划层依然强硬地要求深入 0.28m
            
            x[0] = x0 + A_x * (1 - np.cos(omega_x * phase_t))
            dx[0] = A_x * omega_x * np.sin(omega_x * phase_t)
            ddx[0] = A_x * omega_x**2 * np.cos(omega_x * phase_t)
            
            x[1] = y0 + A_y * (1 - np.cos(omega_y * phase_t))
            dx[1] = A_y * omega_y * np.sin(omega_y * phase_t)
            ddx[1] = A_y * omega_y**2 * np.cos(omega_y * phase_t)
            
        return x, dx, ddx

class AdmittanceFilter:
    """
    【外环算法层】导纳滤波
    功能：根据末端受到的外力，将刚性目标轨迹 x_d 软化为柔顺参考轨迹 x_c
    """
    def __init__(self, dt):
        self.dt = dt
        
        # 导纳参数 (相当于给末端虚拟出的质量、阻尼和弹簧)
        self.Md = np.diag([2.0, 2.0, 2.0])   # 虚拟质量 (kg)
        self.Bd = np.diag([80.0, 80.0, 80.0]) # 虚拟阻尼 (Ns/m)
        self.Kd = np.diag([600.0, 600.0, 600.0]) # 虚拟刚度 (N/m)
        
        self.Md_inv = np.linalg.inv(self.Md)
        
        # 柔顺参考状态 (Compliant state)
        self.x_c = None
        self.dx_c = np.zeros(3)
        
    def filter(self, x_d, dx_d, ddx_d, F_ext, x_curr):
        # 首次运行时，初始化柔顺轨迹对齐当前位置
        if self.x_c is None:
            self.x_c = x_curr.copy()
            
        # 导纳核心微分方程：Md*ddot{e} + Bd*dot{e} + Kd*e = F_ext, e = x_c - x_d
        # 求解柔顺加速度 ddx_c
        error = self.x_c - x_d
        derror = self.dx_c - dx_d
        ddx_c = ddx_d + self.Md_inv @ (F_ext - self.Bd @ derror - self.Kd @ error)
        
        # 欧拉数值积分，更新柔顺速度和位置
        self.dx_c += ddx_c * self.dt
        self.x_c += self.dx_c * self.dt
        
        return self.x_c.copy(), self.dx_c.copy(), ddx_c.copy()

class OperationalSpaceController:
    """
    【内环执行层】操作空间纯位置控制器
    """
    def __init__(self, model, data, ee_name="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        
        self.jacp = np.zeros((3, self.model.nv))
        self.jacp_dot = np.zeros((3, self.model.nv))
        self.M = np.zeros((self.model.nv, self.model.nv))
        
    def compute_torque(self, x_tar, dx_tar, ddx_tar, Kp, Kd) -> tuple:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        hand_pos = self.data.xpos[self.ee_id]
        hand_mat = self.data.xmat[self.ee_id].reshape(3, 3)
        tcp_offset_local = np.array([0.0, 0.0, 0.1034])
        x_curr = hand_pos + hand_mat @ tcp_offset_local
        
        mujoco.mj_jac(self.model, self.data, self.jacp, None, x_curr, self.ee_id)
        J_p = self.jacp[:, :7]
        
        dq_curr = self.data.qvel[:7]
        dx_curr = J_p @ dq_curr
        
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        M_7 = self.M[:7, :7]
        M_inv = np.linalg.inv(M_7)
        Lambda = np.linalg.inv(J_p @ M_inv @ J_p.T)
        
        mujoco.mj_jacDot(self.model, self.data, self.jacp_dot, None, x_curr, self.ee_id)
        J_dot_q_dot = self.jacp_dot[:, :7] @ dq_curr
        
        e = x_tar - x_curr
        de = dx_tar - dx_curr
        
        tau_spring = J_p.T @ (Lambda @ (Kp * e))
        F_cmd = Lambda @ (Kp * e + Kd * de + ddx_tar - J_dot_q_dot)
        tau_task = J_p.T @ F_cmd
        
        I = np.eye(7)
        N_T = I - J_p.T @ Lambda @ J_p @ M_inv
        
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        q_curr = self.data.qpos[:7]
        
        Kp_null = 50.0
        Kd_null = 5.0
        tau_posture = Kp_null * (q_home - q_curr) - Kd_null * dq_curr
        tau_null = N_T @ tau_posture
        
        tau_final = tau_task + tau_null + self.data.qfrc_bias[:7]
        
        return tau_final, x_curr, J_p

class FrankaSimNode:
    """仿真循环"""
    def __init__(self, xml_path="scene.xml"):
        # 当前脚本所在文件夹
        cur_script_dir = Path(__file__).resolve().parent
        # 拼接xml完整路径
        full_xml_path = cur_script_dir / xml_path

        if not full_xml_path.exists():
            raise FileNotFoundError(f"XML 文件不存在: {full_xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(full_xml_path))
        self.data = mujoco.MjData(self.model)
        
        self.controller = OperationalSpaceController(self.model, self.data)
        self.admittance = AdmittanceFilter(dt=self.model.opt.timestep)
        
        # ！！极其重要：导纳控制的内环必须极其僵硬，不能有丝毫柔顺！！
        self.Kp_inner = np.array([5000.0, 5000.0, 5000.0])
        self.Kd_inner = np.array([140.0, 140.0, 140.0])
        
        self._reset_home_pose()
        
    def _reset_home_pose(self):
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = q_home
        mujoco.mj_forward(self.model, self.data)
        
    def run(self):
        print("启动导纳控制仿真")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            real_start_time = time.time()
            
            while viewer.is_running():
                real_elapsed = time.time() - real_start_time
                
                while self.data.time < real_elapsed:
                    t = self.data.time
                    
                    # 1. 规划层：获取绝对目标轨迹 x_d
                    x_d, dx_d, ddx_d = TrajectoryGenerator.get_trajectory(t)
                    
                    # 2. 虚拟六维力矩传感器：通过伪逆读取接触力 F_ext
                    # F_ext = (J^T)^+ * tau_env
                    tau_env = self.data.qfrc_constraint[:7]
                    hand_pos = self.data.xpos[self.controller.ee_id]
                    hand_mat = self.data.xmat[self.controller.ee_id].reshape(3, 3)
                    x_curr = hand_pos + hand_mat @ np.array([0.0, 0.0, 0.1034])
                    
                    mujoco.mj_jac(self.model, self.data, self.controller.jacp, None, x_curr, self.controller.ee_id)
                    J_p = self.controller.jacp[:, :7]
                    
                    # 使用伪逆计算末端3维接触力
                    J_T_pinv = np.linalg.pinv(J_p.T)
                    F_ext = J_T_pinv @ tau_env
                    
                    # 3. 外环 (导纳层)：将 x_d 滤为柔顺的 x_c
                    x_c, dx_c, ddx_c = self.admittance.filter(x_d, dx_d, ddx_d, F_ext, x_curr)
                    
                    # 4. 内环 (控制层)：追踪 x_c
                    tau, _, _ = self.controller.compute_torque(x_c, dx_c, ddx_c, self.Kp_inner, self.Kd_inner)
                    
                    self.data.ctrl[:7] = tau
                    mujoco.mj_step(self.model, self.data)
                    
                    if t > 5.0 and int(t / self.model.opt.timestep) % 500 == 0:
                        print(f"[验证] Time: {t:.2f}s")
                        print(f"原生轨迹 (x_d_z)  : {x_d[2]:.4f} m")
                        print(f"导纳妥协轨迹 (x_c_z)  : {x_c[2]:.4f} m")
                        print(f"真实末端位置 (x_curr_z): {x_curr[2]:.4f} m")
                        print(f"传感器读数 (F_ext_z) : {F_ext[2]:.2f} N")
                        print("-" * 50)
                    
                viewer.sync()

if __name__ == "__main__":
    sim_node = FrankaSimNode()
    sim_node.run()