import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.algo.config import Config
from src.algo import ArmMeasurement
from src.algo import LegMeasurement
from src.algo import ShoulderMeasurement
from src.algo import WaistMeasurement
from src.algo import BodyMeasurement

config = Config(
    front_rgb=Path("lzh_v3/front/Color_1777481144643_1.raw"),
    front_depth=Path("lzh_v3/front/Depth_1777481144608_0.raw"),
    front_ply=Path("lzh_v3/front/PointCloud_Astra Pro_20260430_004541.ply"),
    back_rgb=Path("lzh_v3/back/Color_1777481105969_1.raw"),
    back_depth=Path("lzh_v3/back/Depth_1777481105906_0.raw"),
    back_ply=Path("lzh_v3/back/PointCloud_Astra Pro_20260430_004501.ply"),
    pose_model=Path("model/yolo26n-pose.pt"),
)

model = YOLO(str(config.pose_model))
arm_measurement = ArmMeasurement(config, model)
leg_measurement = LegMeasurement(config, model)
shoulder_measurement = ShoulderMeasurement(config, model)
waist_measurement = WaistMeasurement(config, model)
body_measurement = BodyMeasurement(config, model)

result = body_measurement.run()
print(result)
