# 前端与本地采集上传流程说明

本文档说明智能健身房人体围度测量项目的推荐整体流程。当前目录里的 FastAPI 后端只负责接收数据、排队执行测量脚本、返回测量结果；前端网页和本地采集上传功能建议放到新的目录或新项目里实现。

## 1. 推荐整体架构

```text
前端网页
  -> 用户点击拍摄
  -> 调用本地采集助手

本地采集助手
  -> 控制 RGBD 相机采集 front/back
  -> 本地保存 rgb.raw、depth.raw、pointcloud.ply
  -> 自动打包 zip
  -> 上传到测量后端

FastAPI 测量后端
  -> 解压 zip
  -> 执行 arm / shoulder / leg / waist 四个测量脚本
  -> 返回 job_id
  -> 前端轮询 job 状态并展示结果
```

推荐把系统拆成三个部分：

- `frontend`：浏览器网页，负责交互、触发拍摄、展示测量状态和结果。
- `local-capture-uploader`：运行在本机的小服务或桌面程序，负责相机采集、本地文件保存、zip 打包、上传后端。
- `gym_backend`：当前 FastAPI 测量后端，负责测量任务和结果查询。

## 2. 为什么需要本地采集助手

普通浏览器网页不能静默读取电脑本地文件夹，也不能直接访问任意本地路径，这是浏览器安全限制。

所以如果想实现“点击拍摄后自动保存并上传”，推荐增加一个本地采集助手。前端网页只需要调用本地助手的 HTTP 接口，例如：

```text
POST http://127.0.0.1:7000/capture-and-upload
```

本地助手收到请求后完成：

1. 调用 RGBD 相机采集数据。
2. 保存 front/back 两组文件。
3. 打包成 zip。
4. 上传到测量后端。
5. 把测量后端返回的 `job_id` 返回给前端。

## 3. 文件保存与 zip 结构

本地采集助手建议保存成下面的结构：

```text
capture_bundle/
  front/
    rgb.raw
    depth.raw
    pointcloud.ply
  back/
    rgb.raw
    depth.raw
    pointcloud.ply
```

然后打包为：

```text
capture_bundle.zip
```

后端会自动识别：

- `front/rgb.raw`
- `front/depth.raw`
- `front/*.ply`
- `back/rgb.raw`
- `back/depth.raw`
- `back/*.ply`

实际文件名可以不是完全固定，例如：

```text
front/Color_1777431096104_1.raw
front/Depth_1777431096047_0.raw
front/PointCloud_Astra Pro_20260429_105134.ply
back/Color_1777431146855_1.raw
back/Depth_1777431146727_0.raw
back/PointCloud_Astra Pro_20260429_105230.ply
```

只要路径里包含 `front` / `back`，并且 raw 文件名能区分 `Color/RGB` 和 `Depth`，后端就可以识别。

## 4. 测量后端接口

测量后端默认运行地址：

```text
http://127.0.0.1:8000
```

如果后端部署到另一台服务器，需要替换成服务器地址。

### 4.1 健康检查

```http
GET /health
```

用途：

- 检查后端是否启动。
- 检查四个测量脚本是否存在。
- 检查 `yolo26n-pose.pt` 和 `yolo26n-seg.pt` 是否存在。

示例：

```bash
curl http://127.0.0.1:8000/health
```

### 4.2 上传 zip 并创建测量任务

```http
POST /api/v1/measurements/archive
```

请求类型：

```text
multipart/form-data
```

表单字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `archive` | file | 是 | 无 | 包含 front/back 数据的 zip |
| `measurements` | string | 否 | `all` | 可填 `all` 或 `arm,shoulder,leg,waist` 的任意组合 |
| `arm_side` | string | 否 | `both` | 可填 `left`、`right`、`both` |
| `leg_side` | string | 否 | `both` | 可填 `left`、`right`、`both` |
| `pose_model` | string | 否 | `yolo26n-pose.pt` | 通常不用传 |
| `rgb_width` | int | 否 | `640` | raw RGB 宽度 |
| `rgb_height` | int | 否 | `480` | raw RGB 高度 |
| `rgb_format` | string | 否 | `rgb8` | 可填 `rgb8`、`bgr8`、`gray8` |
| `depth_width` | int | 否 | `640` | raw depth 宽度 |
| `depth_height` | int | 否 | `480` | raw depth 高度 |
| `depth_dtype` | string | 否 | `uint16` | 可填 `uint16`、`uint8`、`float32` |
| `depth_endian` | string | 否 | `little` | 可填 `little`、`big` |
| `depth_scale` | float | 否 | `0.001` | 深度转米比例 |
| `depth_window` | int | 否 | `5` | 深度采样窗口，必须是正奇数 |

curl 示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/measurements/archive" \
  -F "archive=@capture_bundle.zip" \
  -F "measurements=all"
```

成功响应：

```json
{
  "job_id": "99d689aed0114ceb900e283ad97d870a",
  "status": "queued",
  "measurements": ["arm", "shoulder", "leg", "waist"],
  "job_url": "/api/v1/jobs/99d689aed0114ceb900e283ad97d870a"
}
```

前端或本地采集助手需要保存这个 `job_id`。

### 4.3 查询任务状态和结果

```http
GET /api/v1/jobs/{job_id}
```

示例：

```bash
curl http://127.0.0.1:8000/api/v1/jobs/99d689aed0114ceb900e283ad97d870a
```

重点字段：

```json
{
  "job_id": "99d689aed0114ceb900e283ad97d870a",
  "status": "running",
  "summary": {},
  "summary_mm": {},
  "error": null
}
```

`status` 可能值：

- `queued`：任务已创建，还在排队。
- `running`：正在执行测量脚本。
- `succeeded`：全部测量完成。
- `failed`：任务失败。

测量成功后重点读取：

```json
"summary_mm": {
  "arm": {
    "left_arm_total_visible_arc_length_mm": 123.456
  },
  "shoulder": {
    "shoulder_width_mean_mm": 456.789
  },
  "leg": {
    "left_thigh_total_visible_arc_length_mm": 321.123
  },
  "waist": {
    "waist_total_visible_arc_length_mm": 789.123
  }
}
```

如果失败，重点读取：

```json
"error": "...",
"module_results": {
  "arm": {
    "status": "failed",
    "stderr": "..."
  }
}
```

### 4.4 下载结果产物

```http
GET /api/v1/jobs/{job_id}/artifacts/{measurement}/{filename}
```

示例：

```text
GET /api/v1/jobs/99d689aed0114ceb900e283ad97d870a/artifacts/arm/pose_joints.json
```

`measurement` 可选：

- `arm`
- `shoulder`
- `leg`
- `waist`

可下载的文件名可以从任务返回的：

```json
"module_results": {
  "arm": {
    "artifacts": ["pose_joints.json", "...png"]
  }
}
```

中读取。

## 5. 前端网页需要做的任务

前端网页建议负责：

1. 显示拍摄页面。
2. 用户点击“开始拍摄”。
3. 调用本地采集助手接口。
4. 显示采集中、上传中、测量中状态。
5. 拿到 `job_id` 后轮询测量后端。
6. 当 `status=succeeded` 时展示 `summary_mm`。
7. 当 `status=failed` 时展示失败原因，并提示重新采集。

前端不建议直接负责：

- 控制 RGBD 相机底层 SDK。
- 静默读取本地任意文件夹。
- 在浏览器里处理大型 raw/ply 文件。

## 6. 本地采集助手需要做的任务

本地采集助手建议负责：

1. 暴露一个本地 HTTP 接口给前端调用。
2. 控制 RGBD 相机完成 front/back 采集。
3. 将数据保存到本地临时目录。
4. 打包 zip。
5. 调用测量后端 `/api/v1/measurements/archive`。
6. 返回后端的 `job_id` 给前端。

建议本地助手接口：

```http
POST http://127.0.0.1:7000/capture-and-upload
```

请求示例：

```json
{
  "measurements": "all",
  "backend_url": "http://127.0.0.1:8000"
}
```

响应示例：

```json
{
  "job_id": "99d689aed0114ceb900e283ad97d870a",
  "job_url": "http://127.0.0.1:8000/api/v1/jobs/99d689aed0114ceb900e283ad97d870a"
}
```

## 7. 前端轮询示例

```js
async function pollJob(backendUrl, jobId) {
  while (true) {
    const response = await fetch(`${backendUrl}/api/v1/jobs/${jobId}`);
    const job = await response.json();

    if (job.status === "succeeded") {
      return job.summary_mm;
    }

    if (job.status === "failed") {
      throw new Error(job.error || "Measurement failed");
    }

    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}
```

## 8. 前端调用本地助手示例

```js
async function captureAndMeasure() {
  const localHelperUrl = "http://127.0.0.1:7000";
  const backendUrl = "http://127.0.0.1:8000";

  const captureResponse = await fetch(`${localHelperUrl}/capture-and-upload`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      measurements: "all",
      backend_url: backendUrl,
    }),
  });

  if (!captureResponse.ok) {
    throw new Error("Capture and upload failed");
  }

  const { job_id } = await captureResponse.json();
  const result = await pollJob(backendUrl, job_id);
  return result;
}
```

## 9. 直接由前端上传文件夹的备选方案

如果暂时不做本地采集助手，也可以让用户在网页里手动选择文件夹，然后前端上传到：

```http
POST /api/v1/measurements/folder
```

但这种方式需要用户手动选择目录，不能实现完全静默自动上传。

HTML 示例：

```html
<input id="folderInput" type="file" webkitdirectory multiple />
```

JavaScript 示例：

```js
async function uploadFolder(files) {
  const formData = new FormData();
  formData.append("measurements", "all");

  for (const file of files) {
    formData.append("files", file, file.webkitRelativePath || file.name);
  }

  const response = await fetch("http://127.0.0.1:8000/api/v1/measurements/folder", {
    method: "POST",
    body: formData,
  });

  return response.json();
}
```

## 10. 推荐开发顺序

1. 保持当前 FastAPI 后端运行稳定。
2. 新建前端项目，先做结果查询和展示页面。
3. 新建本地采集助手项目，先 mock 采集文件并自动上传 zip。
4. 接入真实 RGBD 相机 SDK。
5. 前端调用本地采集助手，形成完整闭环。

