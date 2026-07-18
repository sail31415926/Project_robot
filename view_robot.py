import mujoco
import mujoco.viewer

# 1. 加载“舞台”文件（它会自动包含 panda.xml）
model = mujoco.MjModel.from_xml_path('scene.xml')

# 2. 生成物理状态数据
data = mujoco.MjData(model)

# 3. 启动 3D 渲染器
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 步进物理仿真
        mujoco.mj_step(model, data)
        
        # 同步画面
        viewer.sync()