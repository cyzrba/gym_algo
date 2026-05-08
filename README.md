# Gym Backend

基于 FastAPI 的智能健身房人体围度测量后端。

接收 RGB、Depth 和点云 PLY 数据，通过 YOLO 姿态检测和点云分析，测量臂围、腿围、腰围和肩宽。

## 技术栈

- Python 3.11+
- FastAPI + Uvicorn
- SQLModel + SQLAlchemy + aiosqlite
- NumPy / OpenCV / SciPy / Matplotlib
- Ultralytics YOLO
- uv

## 安装

```bash
uv sync
```

## 启动

```bash
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 API 文档：

```
http://127.0.0.1:8000/docs
```

## 项目结构

```text
gym_algo  /
├── src/
│   ├── app/                        # FastAPI 应用层
│   │   ├── main.py                 # 应用入口与路由注册
│   │   ├── config.py               # 全局配置（路径、数据库连接）
│   │   ├── api/                    # 接口层
│   │   │   ├── measurement.py      # 测量任务接口（上传并执行测量）
│   │   │   └── task.py             # 任务 CRUD 接口
│   │   ├── model/                  # 数据库模型
│   │   │   └── task.py             # Task 表定义
│   │   ├── schemas/                # Pydantic / SQLModel 数据结构
│   │   │   ├── common.py           # 通用 schema
│   │   │   ├── measurements.py     # 测量结果 schema
│   │   │   └── task.py             # 任务状态与响应 schema
│   │   ├── services/               # 业务逻辑层
│   │   │   ├── measurement.py      # 测量处理服务
│   │   │   └── task.py             # 任务状态管理服务
│   │   └── utils/                  # 工具模块
│   │       ├── constants.py        # COCO 关键点名称等常量
│   │       ├── database.py         # 数据库引擎与会话管理
│   │       └── file_utils.py       # 文件操作工具
│   └── algo/                       # 测量算法层
│       ├── base.py                 # 算法基类（数据加载、姿态检测、点云处理）
│       ├── body.py                 # 全身测量调度器
│       ├── arm.py                  # 臂围测量算法
│       ├── leg.py                  # 腿围测量算法
│       ├── waist.py                # 腰围测量算法
│       ├── shoulder.py             # 肩宽测量算法
│       ├── config.py               # 算法参数配置
│       └── model/                  # 模型文件
│           ├── model.py            # YOLO 模型加载
│           └── yolo26n-pose.pt     # YOLO 姿态检测权重
├── test/                           # 测试
│   ├── test_algo.py                # 算法测试
│   ├── test_task.py                # 任务接口测试
│   └── seed_tasks.py               # 测试数据初始化
├── data/                           # 运行时数据（SQLite 数据库）
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

## 目录说明

- **`src/app/`** — FastAPI 应用层，负责 HTTP 接口、任务管理和数据持久化
- **`src/app/api/`** — 路由接口，`task.py` 管理任务生命周期，`measurement.py` 处理测量请求
- **`src/app/model/`** — SQLModel 数据库表定义
- **`src/app/schemas/`** — 请求/响应数据结构定义
- **`src/app/services/`** — 业务逻辑，协调接口层与算法层
- **`src/algo/`** — 测量算法核心，基于 YOLO 姿态检测 + 点云切片分析
- **`src/algo/model/`** — YOLO 模型权重文件
- **`test/`** — 单元测试与集成测试
- **`data/`** — SQLite 数据库文件存储目录

## API 接口

### 任务管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/tasks` | 创建测量任务 |
| `GET` | `/api/v1/tasks/{task_id}` | 查询任务状态与结果 |
| `PUT` | `/api/v1/tasks/{task_id}` | 更新任务测量结果 |
| `PATCH` | `/api/v1/tasks/{task_id}/status` | 更新任务状态 |
| `DELETE` | `/api/v1/tasks/{task_id}` | 删除任务 |

### 测量执行

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/measurements` | 上传数据并执行测量 |

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务健康检查 |

## 任务状态流转

```text
pending → processing → success
                    → fail
                    → cancel
pending → cancel
```

## 测量流程

```text
创建任务 (POST /api/v1/tasks)
    →
上传数据并执行测量 (POST /api/v1/measurements)
    →
解压 zip，加载前后视角 RGB/Depth/PLY
    →
YOLO 姿态检测 → 关键点提取
    →
点云切片分析 → 臂围/腿围/腰围/肩宽
    →
结果写入 SQLite
    →
轮询任务状态 (GET /api/v1/tasks/{task_id})
```

## 上传数据格式

推荐使用 zip 上传，结构如下：

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

文件名可自定义，但路径中需包含 `front` / `back` 以区分视角，文件名需能区分 RGB 和 Depth。

## curl 示例

创建任务：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/tasks"
```

上传数据并执行测量：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/measurements" \
  -F "task_id={task_id}" \
  -F "archive=@capture_bundle.zip"
```

查询任务：

```bash
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}"
```

## 测试

```bash
uv run pytest
```
