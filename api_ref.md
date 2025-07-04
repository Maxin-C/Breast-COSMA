-----

# API Usage Documentation for Breast Cosma Database

This document provides a comprehensive guide for frontend developers to interact with the Breast Cosma Database API. It covers the available endpoints, HTTP methods, request formats, and example responses for **CRUD (Create, Read, Update, Delete)** operations across all data models.

-----

## Base URL

The base URL for all API endpoints is: `http://0.0.0.0:8000` (or `http://localhost:8000` if running locally).

-----

## Authentication

Currently, no authentication is implemented. All endpoints are publicly accessible. In a production environment, proper authentication and authorization mechanisms should be added.

-----

## Error Handling

The API returns standard HTTP status codes to indicate the success or failure of a request.

  * **`200 OK`**: The request was successful (for GET, PUT, DELETE).
  * **`201 Created`**: The resource was successfully created (for POST).
  * **`400 Bad Request`**: The request was malformed, or validation failed. Details are provided in the `errors` field of the JSON response.
  * **`404 Not Found`**: The requested resource could not be found.
  * **`500 Internal Server Error`**: An unexpected error occurred on the server.

Error responses will typically be in the following JSON format:

```json
{
  "message": "Error message description",
  "errors": {
    "field_name": ["Error details for this field"],
    "another_field": ["Another error"]
  }
}
```

-----

## Date and Time Formats

All date and time fields in API requests and responses should adhere to the following formats:

  * **Date**: `YYYY-MM-DD` (e.g., `2023-10-27`)
  * **Time**: `HH:MM:SS` (e.g., `14:30:00`)
  * **DateTime**: `YYYY-MM-DD HH:MM:SS` (e.g., `2023-10-27 14:30:00`)

-----

## General API Design Notes

  * **`GET` requests** typically return a list of objects or a single object.
  * **`POST` requests** are used to create new resources. Send data in the request body as **JSON**.
  * **`PUT` requests** are used to update existing resources. Send data in the request body as **JSON**. The `ID` in the URL specifies which resource to update.
  * **`DELETE` requests** remove a resource.
  * **Search Endpoints**: All models have a `/search` endpoint supporting filtering by any field. Use query parameters `field` and `value`.

-----

## API Endpoints Reference

### 1\. Users

Manages user accounts.

  * **Model Fields:** `user_id` (int, auto-increment), `wechat_openid` (string), `srrsh_id` (int), `name` (string, **required**), `phone_number` (string), `registration_date` (datetime), `last_login_date` (datetime)

#### List all users

  * **Endpoint**: `/users`
  * **Method**: `GET`
  * **Response**: `200 OK`
    ```json
    [
      {
        "last_login_date": "2023-10-26 10:00:00",
        "name": "Jane Doe",
        "phone_number": "9876543210",
        "registration_date": "2023-01-15 09:00:00",
        "srrsh_id": 1002,
        "user_id": 2,
        "wechat_openid": "openid_jane"
      }
    ]
    ```

#### Get a single user by ID

  * **Endpoint**: `/users/<int:user_id>`
  * **Method**: `GET`
  * **Response**: `200 OK`
    ```json
    {
      "last_login_date": "2023-10-26 10:00:00",
      "name": "Jane Doe",
      "phone_number": "9876543210",
      "registration_date": "2023-01-15 09:00:00",
      "srrsh_id": 1002,
      "user_id": 2,
      "wechat_openid": "openid_jane"
    }
    ```

#### Search users by field

  * **Endpoint**: `/users/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/users/search?field=name&value=John%20Doe`
  * **Response**: `200 OK` (list of matching users) or `404 Not Found`

#### Add a new user

  * **Endpoint**: `/users`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "name": "New User",
      "wechat_openid": "openid_new",
      "srrsh_id": 1003,
      "phone_number": "1122334455",
      "registration_date": "2024-05-01 10:30:00",
      "last_login_date": "2024-05-01 10:30:00"
    }
    ```
  * **Response**: `201 Created`
    ```json
    {
      "message": "User added successfully!",
      "user": {
        "last_login_date": "2024-05-01 10:30:00",
        "name": "New User",
        "phone_number": "1122334455",
        "registration_date": "2024-05-01 10:30:00",
        "srrsh_id": 1003,
        "user_id": 3,
        "wechat_openid": "openid_new"
      }
    }
    ```

#### Edit an existing user

  * **Endpoint**: `/users/<int:user_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`
    ```json
    {
      "name": "Updated User Name",
      "phone_number": "1234567890"
    }
    ```
  * **Response**: `200 OK`
    ```json
    {
      "message": "User updated successfully!",
      "user": {
        "last_login_date": "2024-05-01 10:30:00",
        "name": "Updated User Name",
        "phone_number": "1234567890",
        "registration_date": "2024-05-01 10:30:00",
        "srrsh_id": 1003,
        "user_id": 3,
        "wechat_openid": "openid_new"
      }
    }
    ```

#### Delete a user

  * **Endpoint**: `/users/<int:user_id>`
  * **Method**: `DELETE`
  * **Response**: `200 OK`
    ```json
    {
      "message": "User deleted successfully!"
    }
    ```

-----

### 2\. Recovery Plans

Manages predefined recovery plans.

  * **Model Fields:** `plan_id` (int, auto-increment), `plan_name` (string, **required**), `description` (text), `start_date` (date), `end_date` (date)

#### List all recovery plans

  * **Endpoint**: `/recovery_plans`
  * **Method**: `GET`

#### Get a single recovery plan by ID

  * **Endpoint**: `/recovery_plans/<int:plan_id>`
  * **Method**: `GET`

#### Search recovery plans by field

  * **Endpoint**: `/recovery_plans/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/recovery_plans/search?field=plan_name&value=Early%20Rehab`

#### Add a new recovery plan

  * **Endpoint**: `/recovery_plans`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "plan_name": "Post-Surgery Rehabilitation",
      "description": "Comprehensive exercises for post-surgery recovery.",
      "start_date": "2024-06-01",
      "end_date": "2024-08-31"
    }
    ```

#### Edit an existing recovery plan

  * **Endpoint**: `/recovery_plans/<int:plan_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a recovery plan

  * **Endpoint**: `/recovery_plans/<int:plan_id>`
  * **Method**: `DELETE`

-----

### 3\. Exercises

Manages individual exercises within recovery plans.

  * **Model Fields:** `exercise_id` (int, auto-increment), `exercise_name` (string, **required**), `description` (text), `video_url` (string), `image_url` (string), `duration_minutes` (int), `repetitions` (int)

#### List all exercises

  * **Endpoint**: `/exercises`
  * **Method**: `GET`

#### Get a single exercise by ID

  * **Endpoint**: `/exercises/<int:exercise_id>`
  * **Method**: `GET`

#### Search exercises by field

  * **Endpoint**: `/exercises/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/exercises/search?field=exercise_name&value=Arm%20Stretches`

#### Add a new exercise

  * **Endpoint**: `/exercises`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "exercise_name": "Shoulder Blade Squeeze",
      "description": "Strengthens upper back muscles.",
      "video_url": "http://example.com/videos/shoulder_squeeze.mp4",
      "image_url": "http://example.com/images/shoulder_squeeze.jpg",
      "duration_minutes": 5,
      "repetitions": 15
    }
    ```

#### Edit an existing exercise

  * **Endpoint**: `/exercises/<int:exercise_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete an exercise

  * **Endpoint**: `/exercises/<int:exercise_id>`
  * **Method**: `DELETE`

-----

### 4\. User Recovery Plans

Links users to specific recovery plans and tracks their status.

  * **Model Fields:** `user_plan_id` (int, auto-increment), `user_id` (int, **required**), `plan_id` (int, **required**), `assigned_date` (datetime), `status` (string, **required**, choices: `active`, `completed`, `cancelled`)

#### List all user recovery plans

  * **Endpoint**: `/user_recovery_plans`
  * **Method**: `GET`

#### Get a single user recovery plan by ID

  * **Endpoint**: `/user_recovery_plans/<int:user_plan_id>`
  * **Method**: `GET`

#### Search user recovery plans by field

  * **Endpoint**: `/user_recovery_plans/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/user_recovery_plans/search?field=user_id&value=1`

#### Add a new user recovery plan

  * **Endpoint**: `/user_recovery_plans`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "user_id": 1,
      "plan_id": 1,
      "assigned_date": "2024-06-05 09:00:00",
      "status": "active"
    }
    ```

#### Edit an existing user recovery plan

  * **Endpoint**: `/user_recovery_plans/<int:user_plan_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a user recovery plan

  * **Endpoint**: `/user_recovery_plans/<int:user_plan_id>`
  * **Method**: `DELETE`

-----

### 5\. Calendar Schedules

Manages scheduled events for users.

  * **Model Fields:** `schedule_id` (int, auto-increment), `user_id` (int, **required**), `schedule_date` (date, **required**), `schedule_time` (time), `type` (string, **required**, e.g., `appointment`, `exercise`), `event_details` (text), `is_completed` (boolean), `completion_time` (datetime)

#### List all calendar schedules

  * **Endpoint**: `/calendar_schedules`
  * **Method**: `GET`

#### Get a single calendar schedule by ID

  * **Endpoint**: `/calendar_schedules/<int:schedule_id>`
  * **Method**: `GET`

#### Search calendar schedules by field

  * **Endpoint**: `/calendar_schedules/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/calendar_schedules/search?field=schedule_date&value=2024-07-01`
  * **Note**: For `is_completed`, use `true` or `false` (case-insensitive).

#### Add a new calendar schedule

  * **Endpoint**: `/calendar_schedules`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "user_id": 1,
      "schedule_date": "2024-07-10",
      "schedule_time": "10:00:00",
      "type": "Follow-up",
      "event_details": "Doctor's appointment at clinic.",
      "is_completed": false
    }
    ```

#### Edit an existing calendar schedule

  * **Endpoint**: `/calendar_schedules/<int:schedule_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a calendar schedule

  * **Endpoint**: `/calendar_schedules/<int:schedule_id>`
  * **Method**: `DELETE`

-----

### 6\. Recovery Records

Manages overall daily or session-based recovery records for users.

  * **Model Fields:** `record_id` (int, auto-increment), `user_id` (int, **required**), `record_date` (datetime), `notes` (text)

#### List all recovery records

  * **Endpoint**: `/recovery_records`
  * **Method**: `GET`

#### Get a single recovery record by ID

  * **Endpoint**: `/recovery_records/<int:record_id>`
  * **Method**: `GET`

#### Search recovery records by field

  * **Endpoint**: `/recovery_records/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/recovery_records/search?field=record_date&value=2024-07-01%2014:00:00`

#### Add a new recovery record

  * **Endpoint**: `/recovery_records`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "user_id": 1,
      "record_date": "2024-07-03 15:30:00",
      "notes": "Daily exercise session completed successfully."
    }
    ```

#### Edit an existing recovery record

  * **Endpoint**: `/recovery_records/<int:record_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a recovery record

  * **Endpoint**: `/recovery_records/<int:record_id>`
  * **Method**: `DELETE`

-----

### 7\. Recovery Record Details

Records specific details for each exercise performed within a recovery record.

  * **Model Fields:** `record_detail_id` (int, auto-increment), `record_id` (int, **required**), `exercise_id` (int, **required**), `actual_duration_minutes` (int), `actual_repetitions_completed` (int), `brief_evaluation` (string), `evaluation_details` (text), `completion_timestamp` (datetime)

#### List all recovery record details

  * **Endpoint**: `/recovery_record_details`
  * **Method**: `GET`

#### Get a single recovery record detail by ID

  * **Endpoint**: `/recovery_record_details/<int:record_detail_id>`
  * **Method**: `GET`

#### Search recovery record details by field

  * **Endpoint**: `/recovery_record_details/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/recovery_record_details/search?field=exercise_id&value=5`

#### Add a new recovery record detail

  * **Endpoint**: `/recovery_record_details`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "record_id": 1,
      "exercise_id": 3,
      "actual_duration_minutes": 10,
      "actual_repetitions_completed": 20,
      "brief_evaluation": "Good form.",
      "evaluation_details": "Patient maintained correct posture throughout the exercise.",
      "completion_timestamp": "2024-07-03 15:45:00"
    }
    ```

#### Edit an existing recovery record detail

  * **Endpoint**: `/recovery_record_details/<int:record_detail_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a recovery record detail

  * **Endpoint**: `/recovery_record_details/<int:record_detail_id>`
  * **Method**: `DELETE`

-----

### 8\. Message Chats

Manages chat messages between users, assistants, and professionals.

  * **Model Fields:** `message_id` (int, auto-increment), `conversation_id` (string, **required**), `sender_id` (int, **required**), `sender_type` (string, **required**, choices: `user`, `assistant`, `professional`), `receiver_id` (int, **required**), `receiver_type` (string, **required**, choices: `user`, `assistant`, `professional`), `message_text` (text, **required**), `timestamp` (datetime)

#### List all message chats

  * **Endpoint**: `/messages_chat`
  * **Method**: `GET`

#### Get a single message chat by ID

  * **Endpoint**: `/messages_chat/<int:message_id>`
  * **Method**: `GET`

#### Search message chats by field

  * **Endpoint**: `/messages_chat/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/messages_chat/search?field=conversation_id&value=conv_123`

#### Add a new message chat

  * **Endpoint**: `/messages_chat`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "conversation_id": "conv_abc",
      "sender_id": 1,
      "sender_type": "user",
      "receiver_id": 101,
      "receiver_type": "assistant",
      "message_text": "Hello, I have a question about my exercise plan.",
      "timestamp": "2024-07-03 16:00:00"
    }
    ```

#### Edit an existing message chat

  * **Endpoint**: `/messages_chat/<int:message_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a message chat

  * **Endpoint**: `/messages_chat/<int:message_id>`
  * **Method**: `DELETE`

-----

### 9\. Video Slice Images

Stores images extracted from video slices, potentially for AI analysis or visual feedback.

  * **Model Fields:** `image_id` (int, auto-increment), `exercise_id` (int, **required**), `record_id` (int, **required**), `slice_order` (int), `image_path` (string), `timestamp` (datetime)

#### List all video slice images

  * **Endpoint**: `/video_slice_images`
  * **Method**: `GET`

#### Get a single video slice image by ID

  * **Endpoint**: `/video_slice_images/<int:image_id>`
  * **Method**: `GET`

#### Search video slice images by field

  * **Endpoint**: `/video_slice_images/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/video_slice_images/search?field=exercise_id&value=3`

#### Add a new video slice image

  * **Endpoint**: `/video_slice_images`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "exercise_id": 3,
      "record_id": 1,
      "slice_order": 5,
      "image_path": "/path/to/image_slice_005.jpg",
      "timestamp": "2024-07-03 16:15:00"
    }
    ```

#### Edit an existing video slice image

  * **Endpoint**: `/video_slice_images/<int:image_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a video slice image

  * **Endpoint**: `/video_slice_images/<int:image_id>`
  * **Method**: `DELETE`

-----

### 10\. Forms (Generic Forms)

Manages generic forms for various purposes (e.g., surveys, questionnaires).

  * **Model Fields:** `form_id` (int, auto-increment), `form_name` (string, **required**), `form_content` (text, **required**)

#### List all forms

  * **Endpoint**: `/forms`
  * **Method**: `GET`

#### Get a single form by ID

  * **Endpoint**: `/forms/<int:form_id>`
  * **Method**: `GET`

#### Search forms by field

  * **Endpoint**: `/forms/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/forms/search?field=form_name&value=Daily%20Checkup`

#### Add a new form

  * **Endpoint**: `/forms`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "form_name": "Daily Wellness Check",
      "form_content": "{\"questions\": [{\"type\": \"text\", \"question\": \"How are you feeling today?\"}, {\"type\": \"rating\", \"question\": \"Rate your pain level (1-10)\"}]}"
    }
    ```

#### Edit an existing form

  * **Endpoint**: `/forms/<int:form_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a form

  * **Endpoint**: `/forms/<int:form_id>`
  * **Method**: `DELETE`

-----

### 11\. QoL (Quality of Life)

Records Quality of Life assessment scores for users.

  * **Model Fields:** `qol_id` (int, auto-increment), `form_id` (int, **required**), `user_id` (int, **required**), `score` (int, **required**), `level` (string, **required**)

#### List all QoL records

  * **Endpoint**: `/qols`
  * **Method**: `GET`

#### Get a single QoL record by ID

  * **Endpoint**: `/qols/<int:qol_id>`
  * **Method**: `GET`

#### Search QoL records by field

  * **Endpoint**: `/qols/search?field=<field_name>&value=<field_value>`
  * **Method**: `GET`
  * **Example**: `/qols/search?field=user_id&value=1`

#### Add a new QoL record

  * **Endpoint**: `/qols`
  * **Method**: `POST`
  * **Request Body**: `application/json`
    ```json
    {
      "form_id": 1,
      "user_id": 1,
      "score": 85,
      "level": "Excellent"
    }
    ```

#### Edit an existing QoL record

  * **Endpoint**: `/qols/<int:qol_id>`
  * **Method**: `PUT`
  * **Request Body**: `application/json`

#### Delete a QoL record

  * **Endpoint**: `/qols/<int:qol_id>`
  * **Method**: `DELETE`
