import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """
    【规划层】状态机轨迹生成模块
    """
    @staticmethod
    def get_state_machine_trajectory(t: float) -> tuple:
        """
        状态机设计：
        0.0s - 3.0s: 定点保持 - 测试稳态抗重力
        3.0s - 6.0s: Y轴直线平移 - 测试轨迹切换的平滑度
        6.0s 之后:   Z轴下压撞击 - 故意指令其运动到 Z=0.2 (侵入刚性方块 10 厘米)
        """
        # 基础原点
        x0 = 0.5
        y0 = 0.0
        z0 = 0.5
        
        x = np.array([x0, y0, z0])
        dx = np.zeros(3)
        ddx = np.zeros(3)
        
        if t < 3.0:
            # Phase 1: 绝对定点保持
            pass
            
        elif t < 6.0:
            # Phase 2: Y轴直线平滑移动
            phase_t = t - 3.0
            r = 0.15
            omega = np.pi / 3.0 # 3秒走半圈
            
            x[1] = y0 + r * np.sin(omega * phase_t)
            dx[1] = r * omega * np.cos(omega * phase_t)
            ddx[1] = -r * omega**2 * np.sin(omega * phase_t)
            
        else:
            # Phase 3: 阻抗接触测试 (Impedance Test)
            # 让机械臂在 Y 轴保持偏置的同时，强行向 Z=0.2 处下压
            # 由于刚性墙壁在 Z=0.3，机械臂会产生高达 10cm 的虚拟侵入误差！
            phase_t = t - 6.0
            
            x[1] = y0 + 0.15 * np.sin(np.pi) # 锁定在 Y 的末端
            
            # 修改侵入目标为 0.28 (只产生 2cm 压缩量)
            # 衰减幅度从 0.3 改为 0.22 (因为 0.5 - 0.28 = 0.22)
            z_target = 0.28 + 0.22 * np.exp(-1.0 * phase_t)
            z_vel = -0.22 * np.exp(-1.0 * phase_t)
            z_acc = 0.22 * np.exp(-1.0 * phase_t)
            
            x[2] = z_target
            dx[2] = z_vel
            ddx[2] = z_acc
            
        return x, dx, ddx


class OperationalSpaceController:
    """
    【控制层】操作空间控制器 (Operational Space Controller, OSC)
    功能：封装底层数学推导，对外只暴露控制接口
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
        返回: (7维关节力矩向量, 当前末端误差)
        """
        # 1. 刷新状态与几何原点
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # ==========================================
        # 修复核心：TCP (Tool Center Point) 精确标定补偿
        # ==========================================
        # 获取 hand 基座的绝对位置和旋转矩阵
        hand_pos = self.data.xpos[self.ee_id]
        hand_mat = self.data.xmat[self.ee_id].reshape(3, 3)
        
        # Franka 夹爪的指尖，相对于 hand 基座在 Z 轴方向上有约 10.34 厘米的物理延伸
        tcp_offset_local = np.array([0.0, 0.0, 0.1034])
        
        # 通过矩阵乘法，算出【真正接触方块的指尖】在全局空间中的绝对坐标！
        x_curr = hand_pos + hand_mat @ tcp_offset_local
        
        # 2. 将真正的指尖坐标传入 mj_jac，计算指尖的雅可比矩阵
        mujoco.mj_jac(self.model, self.data, self.jacp, None, x_curr, self.ee_id)
        J_p = self.jacp[:, :7]
        
        dq_curr = self.data.qvel[:7]
        dx_curr = J_p @ dq_curr
        
        # 3. 计算笛卡尔空间惯性矩阵 Lambda
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        M_7 = self.M[:7, :7] # 只取前 7 个关节的质量矩阵，后两个是爪子
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
        
        # ==========================================
        # 动态一致性零空间投影 (Null-Space Projection)
        # 目的：在不影响末端压力的前提下，防止手腕折叠和奇点崩溃
        # ==========================================
        # 计算零空间投影矩阵 N^T = I - J^T * Lambda * J * M^-1
        I = np.eye(7)
        N_T = I - J_p.T @ Lambda @ J_p @ M_inv
        
        # 设定一个极其舒适的黄金避奇点姿态
        q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        q_curr = self.data.qpos[:7]
        dq_curr = self.data.qvel[:7]
        
        # 在关节空间施加一个柔和的 PD 弹簧力，试图保持姿态
        Kp_null = 50.0
        Kd_null = 5.0
        tau_posture = Kp_null * (q_home - q_curr) - Kd_null * dq_curr
        
        # 强制投影到零空间(不干扰笛卡尔空间的末端, 即不产生末端加速度和影响末端的环境力)
        tau_null = N_T @ tau_posture
        
        # 6. 综合控制律 = 主任务 + 零空间任务 + 动力学偏置
        tau_final = tau_task + tau_null + self.data.qfrc_bias[:7]
        
        # 此时返回 tau_null，用于数学等式验证
        return tau_final, np.linalg.norm(e), tau_spring, tau_null


class DataLogger:
    """
    【数据层】量化数据记录器
    功能：高频记录仿真数据，离线保存并生成专业级论文图表
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
        
        # 如果数组为空，直接返回，避免绘图崩溃掩盖真实报错
        if len(self.time_data) == 0:
            print("⚠️ 警告：未记录到任何数据，跳过数据保存与绘图。请检查上方是否有逻辑报错！")
            return
        
        print(f"\n📊 正在保存量化数据至 {filename}.csv ...")
        
        # 1. 导出 CSV (用于备用分析或导入 MATLAB)
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
                
        # 2. 绘制工程级图表
        self._plot_curves()

    def _plot_curves(self):
        print("📈 正在生成可视化图表...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(14, 10))
        
        # 添加专属署名
        fig.text(0.98, 0.02, "Copyright (c) [2026] [Wang qianhang]", 
                 ha='right', va='bottom', fontsize=10, color='gray', alpha=0.7)

        times = np.array(self.time_data)
        errors = np.array(self.error_norm_data)
        torques = np.array(self.torque_data)

        # 子图 1：稳态误差收敛曲线
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(times, errors * 1000, color='#d62728', linewidth=2, label="Tracking Error")
        ax1.set_title("Cartesian Space Tracking Error Norm", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Error (mm)", fontsize=12)
        ax1.set_xlim([times[0], times[-1]])
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc="upper right")

        # 子图 2：七轴关节力矩输出曲线 (考察是否饱和)
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
        self.model = mujoco.MjModel.from_xml_path(xml_path)
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
        print("✅ 机械臂位形已初始化。")
        
    def run(self):
        print("🚀 启动仿真... (关闭渲染窗口或按 Ctrl+C 自动生成图表)")
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
                            
                            print(f"[验证] Time: {t:.2f}s")
                            print(f"  -> 主弹簧力矩 (tau_spring): {tau_spring.round(2)}")
                            print(f"  -> 零空间力矩 (tau_null)  : {tau_null.round(2)}")
                            print(f"  -> 环境力矩 (tau_env)   : {tau_env.round(2)}")
                            print(f"  -> 残差范数 (Residual)  : {residual_norm:.6f} Nm")
                            print("-" * 50)
                        
                        # 记录数据
                        self.logger.log_step(t, x_tar, self.data.xpos[self.controller.ee_id], error, tau)
                        
                    viewer.sync()
        except KeyboardInterrupt:
            print("\n🛑 接收到中断信号，正在终止仿真...")
        finally:
            # 无论是因为关闭窗口还是 Ctrl+C 退出，都会安全触发数据保存与绘图
            self.logger.save_and_plot()


if __name__ == "__main__":
    sim_node = FrankaSimNode()
    sim_node.run()