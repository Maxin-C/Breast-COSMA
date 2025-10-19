# 后端接口文档

## 认证 (`auth`)

### 1\. 登录页面

  * **URL**: `/login`
  * **Method**: `GET`
  * **Description**: 渲染登录页面。
  * **Success Response**:
      * **Code**: 200
      * **Content**: `login.html`

### 2\. 登录接口

  * **URL**: `/api/login`
  * **Method**: `POST`
  * **Description**: 处理护士用户的登录请求。
  * **Request Body**:
    ```json
    {
        "username": "...",
        "password": "..."
    }
    ```
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "message": "登录成功！"
        }
        ```
  * **Error Response**:
      * **Code**: 400, 401
      * **Content**:
        ```json
        {
            "error": "请输入用户名和密码" 
        }
        ```
        或
        ```json
        {
            "error": "用户名或密码错误"
        }
        ```

### 3\. 登出

  * **URL**: `/logout`
  * **Method**: `GET`
  * **Description**: 清除会话并重定向到登录页面。
  * **Success Response**:
      * **Code**: 302 (Redirect)
      * **Redirects to**: `/login`

## 主应用 (`main`)

### 1\. 仪表盘页面

  * **URL**: `/`
  * **Method**: `GET`
  * **Description**: 渲染仪表盘页面。
  * **Success Response**:
      * **Code**: 200
      * **Content**: `dashboard.html`

### 2\. 病例页面

  * **URL**: `/cases`
  * **Method**: `GET`
  * **Description**: 渲染病例列表页面。
  * **Success Response**:
      * **Code**: 200
      * **Content**: `cases.html`

### 3\. 病例详情页面

  * **URL**: `/case/<user_id>`
  * **Method**: `GET`
  * **Description**: 渲染特定病例的详细信息页面。
  * **Success Response**:
      * **Code**: 200
      * **Content**: `case_detail.html`

### 4\. 获取仪表盘统计数据

  * **URL**: `/api/dashboard/stats`
  * **Method**: `GET`
  * **Description**: 获取仪表盘所需的统计数据。
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "population_distribution": {
                "未拔管": 10,
                "已拔管": 20
            },
            "evaluation_results": {
                "未评估": 5,
                "已评估": 15
            },
            "quality_of_life": {
                "未评估": 8,
                "良好": 12
            },
            "project_stats": {
                "total_cases": 30,
                "total_videos": 50,
                "total_reports": 45,
                "duration_days": 100
            },
            "miniprogram_usage": [
                {
                    "date": "2023-10-20",
                    "count": 5
                }
            ]
        }
        ```

### 5\. 获取病例数据 (用于DataTables)

  * **URL**: `/api/cases_datatable`
  * **Method**: `GET`
  * **Description**: 为DataTables.net组件提供病例数据，支持服务器端处理。
  * **Query Parameters**:
      * `draw`: `integer`
      * `start`: `integer`
      * `length`: `integer`
      * `search[value]`: `string`
      * `order[0][column]`: `integer`
      * `columns[...][data]`: `string`
      * `order[0][dir]`: `string` (`asc` or `desc`)
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "draw": 1,
            "recordsTotal": 50,
            "recordsFiltered": 25,
            "data": [
                {
                    "id": 1,
                    "name": "...",
                    "case_id": "...",
                    "reg_date": "...",
                    "category": "...",
                    "sessions": 5,
                    "result": "无评估",
                    "user_id": 1
                }
            ]
        }
        ```

### 6\. 获取病例详情API

  * **URL**: `/api/case/<user_id>`
  * **Method**: `GET`
  * **Description**: 获取特定病例的详细信息。
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "name": "...",
            "case_id": "...",
            "records": [
                {
                    "record_id": 1,
                    "record_date": "...",
                    "details": [
                        {
                            "record_detail_id": 1,
                            "completion_timestamp": "...",
                            "evaluation_details": "...",
                            "video_path": "...",
                            "exercise_id": 1
                        }
                    ]
                }
            ]
        }
        ```

### 7\. 提交护士评估

  * **URL**: `/api/case/evaluations`
  * **Method**: `POST`
  * **Description**: 提交护士对智能评估结果的评估。
  * **Request Body**:
    ```json
    {
        "record_detail_id": 1,
        "score": 8,
        "feedback_text": "..."
    }
    ```
  * **Success Response**:
      * **Code**: 201
      * **Content**:
        ```json
        {
            "message": "评估提交成功！"
        }
        ```

## 报告 (`report`)

### 1\. 上半身检测

  * **URL**: `/detect_upper_body`
  * **Method**: `POST`
  * **Description**: 检测上传的图像中是否包含上半身。
  * **Request Body**: `multipart/form-data` with `image` field.
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "is_upper_body_in_frame": true
        }
        ```

### 2\. 上传训练雪碧图

  * **URL**: `/reports/upload_sprites`
  * **Method**: `POST`
  * **Description**: 上传康复训练的雪碧图。
  * **Request Body**: `multipart/form-data` with `files`, `record_id`, and `exercise_id` fields.
  * **Success Response**:
      * **Code**: 201
      * **Content**:
        ```json
        {
            "message": "Successfully uploaded ... sprite sheets for exercise ....",
            "files_saved": [
                "uploads/slices/..."
            ]
        }
        ```

### 3\. 评估训练报告

  * **URL**: `/reports/evaluate`
  * **Method**: `POST`
  * **Description**: 对指定的康复记录和训练项目进行评估。
  * **Request Body**:
    ```json
    {
        "record_id": 1,
        "exercise_id": 1
    }
    ```
  * **Success Response**:
      * **Code**: 201
      * **Content**:
        ```json
        {
            "success": true,
            "detail_id": 1,
            "video_path": "...",
            "evaluation": "..."
        }
        ```

### 4\. 总结并保存报告

  * **URL**: `/reports/<record_id>/summarize`
  * **Method**: `POST`
  * **Description**: 为指定的康复记录生成并保存总结报告。
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "success": true,
            "record_id": 1,
            "summary": "..."
        }
        ```

## 咨询 (`consult`)

### 1\. 发送咨询消息

  * **URL**: `/consult/messages`
  * **Method**: `POST`
  * **Description**: 发送咨询消息，并获取回复。
  * **Request Body**:
    ```json
    {
        "user_id": 1,
        "message": "...",
        "conversation_id": "...", // Optional
        "mode": "consult", // or "followup"
        "end_conversation": false
    }
    ```
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "response": "...",
            "conversation_id": "...",
            "timestamp": "..."
        }
        ```

### 2\. 获取咨询消息

  * **URL**: `/consult/conversations/<conversation_id>/messages`
  * **Method**: `GET`
  * **Description**: 获取指定对话的所有消息。
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        [
            {
                "role": "user",
                "content": "..."
            },
            {
                "role": "assistant",
                "content": "..."
            }
        ]
        ```

### 3\. 获取对话摘要

  * **URL**: `/consult/conversations/<conversation_id>/summarize`
  * **Method**: `GET`
  * **Description**: 获取指定对话的摘要。
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "summary": "..."
        }
        ```

### 4\. 获取用户咨询上下文

  * **URL**: `/consult/user/context`
  * **Method**: `GET`
  * **Description**: 获取用于咨询的用户上下文信息。
  * **Query Parameters**:
      * `user_id`: `integer`
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "user_id": 1,
            "context": "..."
        }
        ```

## CRUD 操作 (`crud`)

此蓝图下的所有接口均为标准的RESTful CRUD操作，用于管理数据库中的各个实体。

  * `GET /<entity>`: 获取所有记录
  * `GET /<entity>/<id>`: 获取单个记录
  * `POST /<entity>`: 创建新记录
  * `PUT /<entity>/<id>`: 更新记录
  * `DELETE /<entity>/<id>`: 删除记录
  * `GET /<entity>/search?field=<field>&value=<value>`: 按字段搜索

**支持的实体 (`<entity>`)**:

  * `users`
  * `recovery_plans`
  * `exercises`
  * `user_recovery_plans`
  * `calendar_schedules`
  * `recovery_records`
  * `recovery_record_details`
  * `chat_history`
  * `video_slice_images`
  * `qols`
  * `nurses`
  * `nurse_evaluations`

## 消息 (`messaging`)

### 1\. 发送通知

  * **URL**: `/api/send_notification`
  * **Method**: `POST`
  * **Description**: 触发给某个用户发送打卡提醒。
  * **Request Body**:
    ```json
    {
        "user_id": 1
    }
    ```
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "message": "发送成功"
        }
        ```

### 2\. 安排通知

  * **URL**: `/api/schedule_notification`
  * **Method**: `POST`
  * **Description**: 接收前端的用户授权，并安排第二天的通知。
  * **Request Body**:
    ```json
    {
        "user_id": 1,
        "template_id": "..."
    }
    ```
  * **Success Response**:
      * **Code**: 201
      * **Content**:
        ```json
        {
            "message": "订阅成功！我们将在明天上午8点提醒您。",
            "scheduled_time": "..."
        }
        ```

### 3\. 检查订阅状态

  * **URL**: `/api/check_subscription_status`
  * **Method**: `GET`
  * **Description**: 检查用户是否已有待处理的、针对某个模板的通知。
  * **Query Parameters**:
      * `user_id`: `integer`
      * `template_id`: `string`
  * **Success Response**:
      * **Code**: 200
      * **Content**:
        ```json
        {
            "isSubscribed": true
        }
        ```