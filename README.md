# Gym Backend

这是一个用于人体围度/体态测量的 FastAPI 后端项目。后端接收前端或采集程序上传的 RGB、Depth、点云 PLY 文件，然后按任务调用手臂、肩宽、腿围、腰围等测量脚本，最终把每个任务的状态、测量结果和可视化产物保存在本地目录中。

## 主要能力

- 支持单视角上传，也支持 front/back 双视角上传。
- 支持直接上传表单文件、上传 zip 包、以及浏览器文件夹上传。
- 支持的测量类型：`arm`、`shoulder`、`leg`、`waist`。
- 使用 YOLO pose/seg 模型辅助人体关键点或分割识别。
- 每个测量任务会生成独立的 `job_id`，可通过接口查询状态和下载结果文件。

## 运行环境

项目使用 Python 3.11+，依赖由 `uv` 管理。

```bash
uv sync
```

启动后端服务：

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

启动后可访问：

```text
http://127.0.0.1:8000
```

## 主要文件结构

```text
gym_backend/
├── main.py                    # FastAPI 后端入口，负责接口、文件接收、任务创建、状态记录和调度测量脚本
├── arm_pointcloud.py          # 手臂测量脚本，根据 RGB/Depth/PLY 和人体关键点计算手臂相关围度
├── shoulder_pointcloud.py     # 肩部/肩宽测量脚本，处理肩部关键点、点云切片和结果可视化
├── leg_pointcloud.py          # 腿部测量脚本，处理大腿等腿部区域的点云切片和围度计算
├── waist_pointcloud.py        # 腰围测量脚本，定位腰部中心并基于点云切片估算腰围
├── pointcloud.py              # 点云查看/调试工具，用于加载、过滤、降采样和显示 PLY 点云
├── yolo26n-pose.pt            # YOLO 姿态识别模型权重
├── yolo26n-seg.pt             # YOLO 分割模型权重
├── pyproject.toml             # 项目依赖和 Python 版本配置
├── uv.lock                    # uv 锁定文件，记录依赖版本
├── FRONTEND_INTEGRATION.md    # 前端接入说明，包含上传方式、接口调用和数据格式说明
├── PRD.md                     # 产品需求说明
├── img/                       # 上传输入文件保存目录，按 job_id 分组
├── result/                    # 测量结果输出目录，按 job_id 和 measurement 分组
├── .venv/                     # 本地虚拟环境目录
└── .gitignore                 # Git 忽略规则
```

## 核心目录说明

### `img/`

上传的原始输入文件会保存在这里：

```text
img/
└── {job_id}/
    ├── single/
    │   ├── rgb.raw
    │   ├── depth.raw
    │   └── pointcloud.ply
    ├── front/
    │   ├── rgb.raw
    │   ├── depth.raw
    │   └── pointcloud.ply
    └── back/
        ├── rgb.raw
        ├── depth.raw
        └── pointcloud.ply
```

实际目录会根据上传模式有所不同。单视角任务通常使用 `single/`，双视角任务通常使用 `front/` 和 `back/`。

### `result/`

测量输出会保存在这里：

```text
result/
└── {job_id}/
    ├── job.json
    ├── arm/
    ├── shoulder/
    ├── leg/
    └── waist/
```

其中：

- `job.json` 保存任务状态、创建时间、测量类型、结果摘要和错误信息。
- `arm/`、`shoulder/`、`leg/`、`waist/` 保存各自测量脚本生成的图片、JSON 或其他产物。

## 接口列表

### 健康检查

```http
GET /health
```

用于确认服务是否正常运行。

### 表单上传测量

```http
POST /api/v1/measurements
```

适合直接上传 RGB、Depth、PLY 文件。支持单视角和 front/back 双视角。

常用表单字段：

- `measurements`：测量类型，支持 `all` 或逗号分隔，例如 `arm,waist`。
- `input_mode`：输入模式，支持 `single` 或 `front_back`。
- `rgb`、`depth`、`ply`：单视角输入。
- `front_rgb`、`front_depth`、`front_ply`：正面输入。
- `back_rgb`、`back_depth`、`back_ply`：背面输入。

### zip 包上传测量

```http
POST /api/v1/measurements/archive
```

适合上传一个包含 `front/`、`back/` 或 `single/` 的压缩包。

推荐 zip 结构：

```text
capture_bundle.zip
├── front/
│   ├── rgb.raw
│   ├── depth.raw
│   └── pointcloud.ply
└── back/
    ├── rgb.raw
    ├── depth.raw
    └── pointcloud.ply
```

### 文件夹上传测量

```http
POST /api/v1/measurements/folder
```

适合浏览器端选择整个文件夹上传。前端需要保留文件相对路径，例如：

```text
front/rgb.raw
front/depth.raw
front/pointcloud.ply
back/rgb.raw
back/depth.raw
back/pointcloud.ply
```

### 查询任务状态

```http
GET /api/v1/jobs/{job_id}
```

返回任务状态、测量结果摘要、输出文件列表和错误信息。

### 下载结果文件

```http
GET /api/v1/jobs/{job_id}/artifacts/{measurement}/{filename}
```

用于下载指定任务、指定测量类型下的输出文件。

### 删除任务

```http
DELETE /api/v1/jobs/{job_id}
```

删除该任务对应的上传文件和结果文件。

## 调用示例

### front/back 双视角上传

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/measurements" \
  -F "measurements=all" \
  -F "input_mode=front_back" \
  -F "front_rgb=@front_rgb.raw" \
  -F "front_depth=@front_depth.raw" \
  -F "front_ply=@front.ply" \
  -F "back_rgb=@back_rgb.raw" \
  -F "back_depth=@back_depth.raw" \
  -F "back_ply=@back.ply"
```

### 单视角上传

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/measurements" \
  -F "measurements=arm,waist" \
  -F "input_mode=single" \
  -F "rgb=@single_rgb.raw" \
  -F "depth=@single_depth.raw" \
  -F "ply=@single.ply"
```

### zip 上传

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/measurements/archive" \
  -F "measurements=all" \
  -F "archive=@capture_bundle.zip"
```

## 任务流程

```text
上传 RGB/Depth/PLY
        |
        v
FastAPI 保存到 img/{job_id}/
        |
        v
创建 result/{job_id}/job.json
        |
        v
后台调用 arm/shoulder/leg/waist 测量脚本
        |
        v
测量脚本生成结果文件和可视化图片
        |
        v
通过 GET /api/v1/jobs/{job_id} 查询结果
```

## 注意事项

- 模型文件 `yolo26n-pose.pt` 和 `yolo26n-seg.pt` 是测量脚本运行的重要依赖，不要随意删除。
- `img/` 和 `result/` 是运行时目录，里面的数据会随任务上传和删除而变化。
- 前端接入细节可查看 `FRONTEND_INTEGRATION.md`。
- 如果上传原始 `.raw` 文件，需要确保 RGB/Depth 的宽高、格式与测量脚本参数一致。
