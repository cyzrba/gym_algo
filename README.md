# Gym Backend

这是一个基于 FastAPI 的智能健身房人体围度测量后端。

后端接收 RGB、Depth 和点云 PLY 数据，创建测量任务，调用四个测量流程
（`arm`、`shoulder`、`leg`、`waist`），并把任务状态、测量结果和可视化产物保存到本地运行目录。

## 技术栈

- Python 3.11+
- FastAPI
- Uvicorn
- uv
- NumPy / OpenCV / SciPy / Matplotlib
- Ultralytics YOLO

## 安装依赖

```bash
uv sync
```

## 启动服务

推荐启动命令：

```bash
uv run uvicorn app.main:app --app-dir src --reload --host 0.0.0.0 --port 8000
```

兼容启动命令：

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000/docs
```

## 项目结构

```text
gym_backend/
|-- src/
|   |-- app/
|   |   |-- main.py
|   |   |-- api/
|   |   |   `-- routes/
|   |   |       |-- health.py
|   |   |       |-- jobs.py
|   |   |       `-- measurements.py
|   |   |-- core/
|   |   |   |-- constants.py
|   |   |   |-- paths.py
|   |   |   `-- validation.py
|   |   |-- schemas/
|   |   |   |-- common.py
|   |   |   `-- measurements.py
|   |   |-- services/
|   |   |   |-- artifact_service.py
|   |   |   |-- input_service.py
|   |   |   |-- job_service.py
|   |   |   `-- measurement_runner.py
|   |   `-- utils/
|   |       `-- file_utils.py
|   `-- pointcloud/
|       |-- pointcloud.py
|       |-- arm_pointcloud.py
|       |-- shoulder_pointcloud.py
|       |-- leg_pointcloud.py
|       `-- waist_pointcloud.py
|-- img/
|-- result/
|-- main.py
|-- pyproject.toml
|-- uv.lock
|-- README.md
|-- .gitignore
|-- .python-version
|-- yolo26n-pose.pt
`-- yolo26n-seg.pt
```

## 目录说明

- `src/app/`
  - FastAPI 应用代码。
  - 包含路由、业务服务、参数校验、路径配置和应用组装。

- `src/app/api/routes/`
  - 接口层。
  - `measurements.py` 负责上传和创建测量任务。
  - `jobs.py` 负责查询任务、删除任务和下载产物。
  - `health.py` 负责健康检查返回内容。

- `src/app/core/`
  - 核心配置层。
  - 管理常量、路径和基础参数校验。

- `src/app/services/`
  - 业务逻辑层。
  - 负责输入保存、zip 解压、任务状态读写、测量脚本调度和产物解析。

- `src/app/schemas/`
  - 数据结构定义。
  - 当前主要使用 dataclass 保存内部参数结构。

- `src/pointcloud/`
  - 点云和人体围度测量脚本。
  - 后端会以 Python module 的方式调用这些脚本。

- `img/`
  - 运行时输入目录。
  - 上传文件会保存到 `img/{job_id}/`。

- `result/`
  - 运行时输出目录。
  - 每个任务会写入 `result/{job_id}/`。

- `main.py`
  - 根目录兼容入口。
  - 内部只负责导入 `src/app/main.py` 里的 FastAPI app。

## 主要接口

- `GET /health`
- `POST /api/v1/measurements`
- `POST /api/v1/measurements/archive`
- `POST /api/v1/measurements/folder`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/artifacts/{measurement}/{filename}`
- `DELETE /api/v1/jobs/{job_id}`

## 任务流程

```text
上传输入数据
    ->
创建 job
    ->
写入 result/{job_id}/job.json
    ->
依次运行 arm / shoulder / leg / waist
    ->
保存 JSON 和图片产物
    ->
通过 GET /api/v1/jobs/{job_id} 轮询结果
```

## zip 上传格式

推荐使用 `POST /api/v1/measurements/archive` 上传 zip。

zip 文件建议结构：

```text
capture_bundle.zip
|-- front/
|   |-- rgb.raw
|   |-- depth.raw
|   `-- pointcloud.ply
`-- back/
    |-- rgb.raw
    |-- depth.raw
    `-- pointcloud.ply
```

实际文件名可以不同，但路径中需要能识别 `front` / `back`，并且 raw 文件名需要能区分 RGB 和 Depth。

例如：

```text
front/Color_1777431096104_1.raw
front/Depth_1777431096047_0.raw
front/PointCloud_Astra Pro_20260429_105134.ply
back/Color_1777431146855_1.raw
back/Depth_1777431146727_0.raw
back/PointCloud_Astra Pro_20260429_105230.ply
```

## curl 示例

上传 zip 并执行全部测量：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/measurements/archive" \
  -F "archive=@capture_bundle.zip" \
  -F "measurements=all"
```

查询任务：

```bash
curl "http://127.0.0.1:8000/api/v1/jobs/{job_id}"
```

## 注意事项

- `img/`、`result/`、`.venv/`、`.mplconfig/` 和模型权重文件默认不提交到 git。
- 运行测量任务前，需要确保本地存在 `yolo26n-pose.pt`。
- `yolo26n-seg.pt` 当前作为后续分割模型扩展预留。
- 当前任务状态保存在本地 `job.json` 文件中，没有接数据库。
