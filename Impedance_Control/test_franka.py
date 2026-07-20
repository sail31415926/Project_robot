import mujoco
import mujoco.viewer
import numpy as np
import time

xml_path = r"scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

ee_body_name = "link7"  
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

Kp_x = np.array([3000.0, 3000.0, 3000.0]) 
Kd_x = np.array([110.0, 110.0, 110.0])

jacp = np.zeros((3, model.nv))
jacp_dot = np.zeros((3, model.nv))
M = np.zeros((model.nv, model.nv))

# 初始化极其重要的黄金准备姿态，避开奇点
q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
data.qpos[:7] = q_home
mujoco.mj_forward(model, data)

print("🚀 启动完美轨迹跟踪 (反馈线性化 + 实时同步)...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    real_start_time = time.time()
    
    while viewer.is_running():
        real_elapsed = time.time() - real_start_time
        
        # 内层物理循环
        while data.time < real_elapsed:
            t = data.time
            dt = model.opt.timestep
            
            x_target = np.array([0.5, 0.1 * np.sin(2.0 * t), 0.4 + 0.1 * np.cos(2.0 * t)])
            dx_target = np.array([0.0, 0.2 * np.cos(2.0 * t), -0.2 * np.sin(2.0 * t)])
            ddx_target = np.array([0.0, -0.4 * np.sin(2.0 * t), -0.4 * np.cos(2.0 * t)])
            
            mujoco.mj_kinematics(model, data) # 更新位置、速度、加速度
            mujoco.mj_comPos(model, data) # 更新质心位置
        
            # 1. 明确 x_current 是几何原点
            x_current = data.xpos[ee_body_id]
            
            # ==========================================
            # 修复 1：强行计算“几何原点 x_current”处的雅可比
            # 废弃 mj_jacBodyCom，改用 mj_jac
            # ==========================================
            mujoco.mj_jac(model, data, jacp, None, x_current, ee_body_id)
            J_p = jacp[:, :7]
            
            dq_current = data.qvel[:7]
            dx_current = J_p @ dq_current
            
            # 2. 提取并计算笛卡尔空间质量惯性矩阵 Lambda (Λ)
            mujoco.mj_fullM(model, M, data.qM)
            M_7 = M[:7, :7]
            M_inv = np.linalg.inv(M_7)
            Lambda = np.linalg.inv(J_p @ M_inv @ J_p.T)
            
            # ==========================================
            # 修复 2：求雅可比导数时，传入的也必须是“几何原点 x_current”
            # ==========================================
            mujoco.mj_jacDot(model, data, jacp_dot, None, x_current, ee_body_id)
            J_dot_q_dot = jacp_dot[:, :7] @ dq_current
            
            e = x_target - x_current
            de = dx_target - dx_current
            
            # ==========================================
            # 修复 3：数学上绝对完美的完全体解耦控制律
            # 将 PD 误差项也全部左乘 Lambda
            # ==========================================
            F_cmd = Lambda @ (Kp_x * e + Kd_x * de + ddx_target - J_dot_q_dot)
            
            tau_task = J_p.T @ F_cmd
            tau_final = tau_task + data.qfrc_bias[:7]
            
            data.ctrl[:7] = tau_final
            
            mujoco.mj_step(model, data)
            
            # 打印逻辑移入内部：保证 t 和 x_target 绝对已被赋值
            if int(t / dt) % 100 == 0:
                error = np.linalg.norm(x_target - x_current)
                print(f"Time: {t:.2f}s | Target: {x_target.round(4)} | Current: {x_current.round(4)} | Error: {error:.6f}")
                
        # 外层渲染刷新
        viewer.sync()