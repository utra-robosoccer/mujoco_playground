import mujoco as mj
import mujoco.viewer  # note: imported separately
import numpy as np
import time, os, cv2

XML_PATH = "scene2.xml"
CAM_NAME = "ball_cam"
CAM_W, CAM_H = 640, 480

model = mj.MjModel.from_xml_path(XML_PATH)
data  = mj.MjData(model)

cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, CAM_NAME)
if cam_id < 0:
    raise RuntimeError(f"Camera '{CAM_NAME}' not found")

renderer = mj.Renderer(model, CAM_H, CAM_W)

def render_cam():
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()

# ------------------------------------------------------------------
# passive viewer – returns immediately and lets us step ourselves
# ------------------------------------------------------------------
viewer = mujoco.viewer.launch_passive(model, data)
cv2.namedWindow("ball_cam", cv2.WINDOW_NORMAL)

data.qvel[0] = 18


while viewer.is_running():
    mj.mj_step(model, data)          # physics step
    rgb = render_cam()               # off-screen render
    cv2.imshow("ball_cam", rgb[:,:,::-1])
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break
    viewer.sync()                    # keep viewer in sync

cv2.destroyAllWindows()
viewer.close()