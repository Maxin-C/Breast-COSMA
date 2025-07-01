import os
import shutil
import time # 导入 time 库
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image # 导入Pillow库

# 初始化 Flask 应用
app = Flask(__name__)

# 配置上传文件的存储路径
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=['GET'])
def none():
    print("S")
    return jsonify({'error': 'Missing file or form data (layout, frames)'}), 200

@app.route("/pose_estimation", methods=['POST'])
def pose_estimation():
    """
    接收小程序上传的雪碧图，切割成多帧后进行处理
    """
    print("="*40)
    print(f"Incoming Request Headers:\n{request.headers}")

    # 1. 检查文件部分
    if 'sprite_image' not in request.files:
        return jsonify({'error': 'No "sprite_image" part in the request'}), 400

    file = request.files['sprite_image']

    # 2. 获取元数据
    layout = request.form.get('layout') # e.g., '4x2'
    total_frames_str = request.form.get('frames') # e.g., '8'
    
    if not all([file.filename, layout, total_frames_str]):
        return jsonify({'error': 'Missing file or form data (layout, frames)'}), 400

    # 【优化】为本次请求创建一个更健壮的唯一会话文件夹
    request_id = f"session_{int(time.time())}_{secure_filename(file.filename)}"
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], request_id)
    os.makedirs(session_folder, exist_ok=True)
    
    try:
        # 3. 保存上传的雪碧图
        sprite_path = os.path.join(session_folder, secure_filename(file.filename))
        file.save(sprite_path)
        print(f"Sprite sheet saved to: {sprite_path}")

        # 5. 【占位符】将切割出的帧图像序列送入AI模型进行推理
        # ==========================================================
        # model_input = preprocess_images(extracted_frames_paths)
        # prediction_result = your_ai_model.predict(model_input)
        # action_label = postprocess_result(prediction_result)
        
        # 模拟模型处理返回一个结果
        action_label = "俯卧撑" # 假设模型识别出了“俯卧撑”动作
        # ==========================================================
        
        # 6. 返回最终结果
        return jsonify({'label': action_label, 'status': 'success'})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500
    
    # finally:
    #     # 7. 清理工作：删除为本次请求创建的整个会话文件夹
    #     if os.path.exists(session_folder):
    #         shutil.rmtree(session_folder)
    #         print(f"Cleaned up session folder: {session_folder}")


if __name__ == '__main__':
    # 按要求将端口设置为 8000
    app.run(host='0.0.0.0', port=8000, debug=True)
