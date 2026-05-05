MEASUREMENTS = ("arm", "shoulder", "leg", "waist")

RGB_FORMATS = {"rgb8", "bgr8", "gray8"}
DEPTH_DTYPES = {"uint16", "uint8", "float32"}
DEPTH_ENDIANS = {"little", "big"}

MEASUREMENT_MODULES = {
    "arm": "pointcloud.arm_pointcloud",
    "shoulder": "pointcloud.shoulder_pointcloud",
    "leg": "pointcloud.leg_pointcloud",
    "waist": "pointcloud.waist_pointcloud",
}
