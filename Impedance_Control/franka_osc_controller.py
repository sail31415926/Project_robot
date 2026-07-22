import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time
import csv
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """
    【规划层】轨迹生成模块(柔顺打磨抛光)
    """
    @staticmethod
    def get_state_machine_trajectory(t: float) -> tuple:
        # 定义打磨的振幅和频率
        A_x, omega_x = 0.04, 0.8 * np.pi
        A_y, omega_y = 0.06, 1.6 * np.pi
        
        # 修复核心：巧妙地向左后方偏移基础原点！
        # 这样 1-cos 产生的 [0, 2A] 轨迹，就会完美居中于 0.5 和 0.0
        x0 = 0.5 - A_x  # 降落点变成 X=0.46
        y0 = 0.0 - A_y  # 降落点变成 Y=-0.06
        z0 = 0.5
        
        x = np.array([x0, y0, z0])
        dx = np.zeros(3)
        ddx = np.zeros(3)
        
        if t < 0.5:
            # Phase 1:悬停0.5秒
            pass
            
        elif t < 2.0:
            # Phase 2:1.5秒内快速下压，建立法向打磨力
            phase_t = t - 0.5
            
            z_target = 0.28 + 0.22 * np.exp(-3.0 * phase_t)
            z_vel = -0.22 * 3.0 * np.exp(-3.0 * phase_t)
            z_acc = 0.22 * 3.0**2 * np.exp(-3.0 * phase_t)
            
            x[2] = z_target
            dx[2] = z_vel
            ddx[2] = z_acc
            
        else:
            # Phase 3:2.0秒之后立即开始居中8字打磨
            phase_t = t - 2.0
            x[2] = 0.28 # 死死锁住法向打磨力
            
            # X/Y轴位置平滑追踪，速度/加速度导数
            x[0] = x0 + A_x * (1 - np.cos(omega_x * phase_t))
            dx[0] = A_x * omega_x * np.sin(omega_x * phase_t)
            ddx[0] = A_x * omega_x**2 * np.cos(omega_x * phase_t)
            
            x[1] = y0 + A_y * (1 - np.cos(omega_y * phase_t))
            dx[1] = A_y * omega_y * np.sin(omega_y * phase_t)
            ddx[1] = A_y * omega_y**2 * np.cos(omega_y * phase_t)
            
        return x, dx, ddx


class OperationalSpaceController:
    """
    控制层：操作空间控制器(OSC)
    """
    def __init__(self, model, data, ee_name="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        
        # 预先分配内存，避免在 1000Hz 的高频控制循环中产生动态内存分配开销
        self.jacp = np.zeros((3, self.model.nv))
        self.jacp_dot = np.zeros((3, self.model.nv))
        self.M = np.zeros((self.model.nv, self.model.nv))
        
    def compute_torque(self, x_tar, dx_tar, ddx_tar, Kp, Kd) -> tuple:
        """
        计算反馈线性化力矩
        返回:(7维关节力矩向量,当前末端误差)
        """
        # 1. 刷新状态与几何原点
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # 获取 hand 基座的绝对位置和旋转矩阵
        hand_pos = self.data.xpos[self.ee_id]
        hand_mat = self.data.xmat[self.ee_id].reshape(3, 3)
        
        # Franka 夹爪的指尖，相对于 hand 基座在 Z 轴方向上有约 10.34 厘米的延伸
        tcp_offset_local = np.array([0.0, 0.0, 0.1034])
        
        # 通过矩阵乘法，算出真正接触方块的指尖在全局空间中的绝对坐标
        x_curr = hand_pos + hand_mat @ tcp_offset_local
        
        # 2. 将真正的指尖坐标传入 mj_jac，计算指尖的雅可比矩阵
        mujoco.mj_jac(self.model, self.data, self.jacp, None, x_curr, self.ee_id)
        J_p = self.jacp[:, :7]
        
        dq_curr = self.data.qvel[:7]
        dx_curr = J_p @ dq_curr
        
        # 3. 计算笛卡尔空间惯性矩阵Lambda
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        M_7 = self.M[:7, :7] # 只取前7个关节的质量矩阵，后两个是爪子
        M_inv = np.linalg.inv(M_7)
        Lambda = np.linalg.inv(J_p @ M_inv @ J_p.T)
        
        # 4. 计算雅可比导数补偿项
        mujoco.mj_jacDot(self.model, self.data, self.jacp_dot, None, x_curr, self.ee_id)
        J_dot_q_dot = self.jacp_dot[:, :7] @ dq_curr
        
        # 5. 反馈线性化核心控制律
        e = x_tar - x_curr
        de = dx_tar - dx_curr
        
        tau_spring = J_p.T @ (Lambda @ (Kp * e))
        F_cmd = Lambda @ (Kp * e + Kd * de + ddx_tar - J_dot_q_dot)
        tau_task = J_p.T @ F_cmd
        
        # 动态一致性零空间投影 目的：在不影响末端压力的前提下，防止手腕折叠和奇点崩溃
        # 计算零空间投影矩阵 N^T = I - J^T * Lambda * J * M^-1
        I = np.eye(7)
        N_T = I - J_p.T @ Lambda @ J_p @ M_inv
        
        # 设定避奇点姿态
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        q_curr = self.data.qpos[:7]
        dq_curr = self.data.qvel[:7]
        
        # 关节空间施加一个柔和的PD弹簧力，保持姿态
        Kp_null = 50.0
        Kd_null = 5.0
        tau_posture = Kp_null * (q_home - q_curr) - Kd_null * dq_curr
        
        # 投影到零空间(不干扰笛卡尔空间的末端, 即不产生末端加速度和影响末端的环境力)
        tau_null = N_T @ tau_posture
        
        # 6. 综合控制\tau = 任务 + 零空间姿态纠正 + 动力学偏置
        tau_final = tau_task + tau_null + self.data.qfrc_bias[:7]
        
        return tau_final, np.linalg.norm(e), tau_spring, tau_null


class DataLogger:
    """
    【数据层】量化数据记录器
    """
    def __init__(self):
        self.time_data = []
        self.error_norm_data = []
        self.target_pos_data = []
        self.current_pos_data = []
        self.torque_data = []
        
    def log_step(self, t, x_tar, x_curr, error, tau):
        """在每个控制周期调用，压入数据"""
        self.time_data.append(t)
        self.target_pos_data.append(x_tar.copy())
        self.current_pos_data.append(x_curr.copy())
        self.error_norm_data.append(error)
        self.torque_data.append(tau.copy())
        
    def save_and_plot(self, filename="franka_tracking_log"):
        """仿真结束时调用，保存 CSV 并绘制图表"""
        
        # 如果数组为空，直接返回
        if len(self.time_data) == 0:
            print("warning：未记录到任何数据")
            return
        
        print(f"\n 保存数据至 {filename}.csv")
        
        # 1. 导出 CSV
        with open(f"{filename}.csv", mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["Time(s)", "Target_X", "Target_Y", "Target_Z", 
                      "Current_X", "Current_Y", "Current_Z", "Error_Norm(m)"] + \
                     [f"Tau_{i+1}(Nm)" for i in range(7)]
            writer.writerow(header)
            for i in range(len(self.time_data)):
                row = [self.time_data[i]] + \
                      self.target_pos_data[i].tolist() + \
                      self.current_pos_data[i].tolist() + \
                      [self.error_norm_data[i]] + \
                      self.torque_data[i].tolist()
                writer.writerow(row)
                
        # 2. 绘制图表
        self._plot_curves()

    def _plot_curves(self):
        print("生成可视化图表...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(14, 10))
        
        # 添加署名
        fig.text(0.98, 0.02, "Copyright (c) [2026] [Wang qianhang]", 
                 ha='right', va='bottom', fontsize=10, color='gray', alpha=0.7)

        times = np.array(self.time_data)
        errors = np.array(self.error_norm_data)
        torques = np.array(self.torque_data)

        # 子图 1：稳态误差收敛
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(times, errors * 1000, color='#d62728', linewidth=2, label="Tracking Error")
        ax1.set_title("Cartesian Space Tracking Error Norm", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Error (mm)", fontsize=12)
        ax1.set_xlim([times[0], times[-1]])
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc="upper right")

        # 子图 2：七轴关节力矩输出
        ax2 = plt.subplot(2, 1, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, 7))
        for i in range(7):
            ax2.plot(times, torques[:, i], color=colors[i], alpha=0.8, label=f"Joint {i+1}")
        ax2.set_title("Control Effort: Joint Torques", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Torque (Nm)", fontsize=12)
        ax2.set_xlim([times[0], times[-1]])
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc="upper right", ncol=7)

        plt.tight_layout()
        plt.show()


class FrankaSimNode:
    """【执行层】：生命周期管理和 DataLogger 挂载"""
    def __init__(self, xml_path="scene.xml"):
        # 当前脚本所在文件夹
        cur_script_dir = Path(__file__).resolve().parent
        # 拼接xml完整路径
        full_xml_path = cur_script_dir / xml_path

        if not full_xml_path.exists():
            raise FileNotFoundError(f"XML 文件不存在: {full_xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(full_xml_path))
        self.data = mujoco.MjData(self.model)
        
        self.controller = OperationalSpaceController(self.model, self.data)# 挂载控制器
        self.logger = DataLogger() # 挂载日志记录器
        
        self.Kp = np.array([3000.0, 3000.0, 1000.0])
        self.Kd = np.array([110.0, 110.0, 110.0])
        
        self._reset_home_pose() # 初始化机械臂位形
        
    def _reset_home_pose(self):
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = q_home
        mujoco.mj_forward(self.model, self.data)
        print("机械臂位形已初始化。")
        
    def run(self):
        print("启动仿真 (Ctrl+C生成图表)")
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                real_start_time = time.time()
                
                while viewer.is_running():
                    real_elapsed = time.time() - real_start_time
                    
                    while self.data.time < real_elapsed:
                        t = self.data.time # 当前仿真时间
                        
                        # 轨迹生成器：获取当前时间下的目标状态
                        x_tar, dx_tar, ddx_tar = TrajectoryGenerator.get_state_machine_trajectory(t)
                        
                        # 注意这里多接收了一个 tau_null
                        tau, error, tau_spring, tau_null = self.controller.compute_torque(x_tar, dx_tar, ddx_tar, self.Kp, self.Kd)
                        
                        self.data.ctrl[:7] = tau
                        mujoco.mj_step(self.model, self.data)
                        
                        if t > 8.0 and int(t / self.model.opt.timestep) % 500 == 0:
                            tau_env = self.data.qfrc_constraint[:7]
                            
                            # 终极物理平衡等式：主任务弹簧力 + 零空间姿态力 + 环境反作用力 = 0
                            residual = tau_spring + tau_null + tau_env
                            residual_norm = np.linalg.norm(residual)
                            
                            print(f"[Time: {t:.2f}s]")
                            print(f"主弹簧力矩 (tau_spring): {tau_spring.round(2)}")
                            print(f"零空间力矩 (tau_null)  : {tau_null.round(2)}")
                            print(f"环境力矩 (tau_env)   : {tau_env.round(2)}")
                            print(f"残差范数 (Residual)  : {residual_norm:.6f} Nm")
                            print("-" * 50)
                        
                        # 记录数据
                        self.logger.log_step(t, x_tar, self.data.xpos[self.controller.ee_id], error, tau)
                        
                    viewer.sync()
        except KeyboardInterrupt:
            print("\n结束仿真")
        finally:
            # 关闭窗口或Ctrl+C退出，都触发数据保存与绘图
            self.logger.save_and_plot()


if __name__ == "__main__":
    sim_node = FrankaSimNode()
    sim_node.run()