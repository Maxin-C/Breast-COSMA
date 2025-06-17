import random
import time

# 预定义的15个康复动作列表
ACTION_LIST = [
    'Hold Loose Fist', 'Rotate Wrist', 'Flex and Extend Forearm',
    'Touch Shoulder', 'Touch Ear', 'Deep Breathing Exercise',
    'Hold Elbow and Comb Hair Exercise', 'Shrug Exercise', 'Torso Twist Exercise',
    'Reach Overhead and Touch Ear', 'Pendulum Exercise', 'Wall Climbing Exercise',
    'Circle Drawing Exercise', 'Pulley Exercise', 'Wash Back Exercise'
]

def mock_pose_estimation_model(frames: list) -> str:
    """
    模拟姿态识别模型 (同步版本)。
    - 使用 time.sleep 来模拟同步IO或CPU密集型任务的耗时。
    """
    print(f"INFO: [模型] 正在识别 {len(frames)} 帧...")
    if len(frames) != 8:
        return "Unknown Action"
    
    time.sleep(0.2)  # 模拟模型推理延迟

    if random.random() < 0.7:
        action = random.choice(ACTION_LIST)
    else:
        action = 'Transitioning'
        
    print(f"INFO: [模型] 识别结果: {action}")
    return action

def mock_qwen_vl_model(frames: list, action: str) -> dict:
    """
    模拟Qwen-VL多模态大模型进行评估 (同步版本)。
    - 使用 time.sleep 来模拟同步IO或CPU密集型任务的耗时。
    """
    print(f"INFO: [大模型] 正在评估动作 '{action}' (共 {len(frames)} 帧)...")
    time.sleep(1.5)  # 模拟大模型推理延迟

    score = random.randint(70, 95)
    issues = [
        {"timestamp_sec": round(random.uniform(1, 5), 1), "issue": "动作幅度不足", "suggestion": "请尽量将手臂伸展到最大程度。"},
        {"timestamp_sec": round(random.uniform(5, 10), 1), "issue": "速度过快", "suggestion": "请放慢动作，感受肌肉的发力。"},
    ]
    
    result = {
        "score": score,
        "feedback": {
            "overall": f"动作 '{action}' 完成度为 {score}%，整体良好，但有几个细节需要注意。",
            "key_issues": random.sample(issues, k=random.randint(1, 2))
        },
        "action_analyzed": action,
        "frame_count": len(frames)
    }
    
    print(f"INFO: [大模型] 评估完成。")
    return result
