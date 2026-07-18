import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. 加载模型与数据
xml_path = r"scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. 找到末端执行器 (End-Effector)
# 在官方模型中，连杆7 (link7) 或 hand 是末端。我们用 body 的名字获取其 ID
ee_body_name = "link7"  
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

# 3. 设定笛卡尔空间的阻抗参数 (X, Y, Z 三个方向的刚度和阻尼)
# 这里的 Kp 变成了空间刚度 (N/m)，Kd 变成了空间阻尼 (Ns/m)
Kp_x = np.array([1000.0, 1000.0, 1000.0]) 
Kd_x = np.array([50.0, 50.0, 50.0])

# 需要提前分配一个二维数组来接收它，存入的是末端执行器的平动雅可比 (3xnv)，其中 nv 是模型的自由度数量
jacp = np.zeros((3, model.nv))

print("🚀 启动笛卡尔空间 3D 阻抗控制...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # 记录起始时间，用于生成圆轨迹
    start_time = time.time()
    
    while viewer.is_running():
        t = time.time() - start_time
        
        # ==========================================
        # 1. 轨迹规划：生成一个空间中的圆形目标轨迹
        # 圆心在 (0.5, 0.0, 0.4)，半径 0.1m
        x_target = np.array([
            0.5, 
            0.1 * np.sin(2.0 * t), 
            0.4 + 0.1 * np.cos(2.0 * t)
        ])
        # ==========================================
        
        # ==========================================
        # 2. 运动学正解与雅可比提取 (MuJoCo 内置 API)
        # 强制 MuJoCo 更新一次运动学链 (计算出所有连杆的位置和雅可比)
        mujoco.mj_kinematics(model, data)# 正向运动学求解 + 刷新所有连杆根部世界位姿
        mujoco.mj_comPos(model, data) # 基于上面算出的连杆位姿，进一步算出每根杆件质心的全局坐标与惯性矩阵
        
        # 获取末端当前实际 3D 位置
        x_current = data.xpos[ee_body_id]
        
        # 提取末端的平动雅可比矩阵 (Translational Jacobian)
        mujoco.mj_jacBodyCom(model, data, jacp, None, ee_body_id)
        # 将一维数组 reshape 为 (3, nv) 的矩阵，并且只取前 7 列 (对应 7 个关节)
        J_p = jacp[:, :7]
        # ==========================================
        
        # ==========================================
        # 3. 笛卡尔空间阻抗控制律解算
        # 获取当前关节速度，通过雅可比矩阵正向映射得到末端空间速度
        dq_current = data.qvel[:7]
        dx_current = J_p @ dq_current # v = J * dq
        
        # 计算空间中的虚拟控制力 F_cmd (大小为 3x1)
        F_cmd = Kp_x * (x_target - x_current) - Kd_x * dx_current
        
        # 核心映射：力矩 = 雅可比转置 * 空间力
        tau_task = J_p.T @ F_cmd
        
        # 叠加重力与科里奥利力补偿
        tau_final = tau_task + data.qfrc_bias[:7]
        # ==========================================
        
        # 下发控制指令
        data.ctrl[:7] = tau_final
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)