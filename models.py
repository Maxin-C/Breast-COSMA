import random
import asyncio
import json

# 预定义的15个康复动作列表
ACTION_LIST = [
    'Hold Loose Fist', 'Rotate Wrist', 'Flex and Extend Forearm', 
    'Touch Shoulder', 'Touch Ear', 'Deep Breathing Exercise', 
    'Hold Elbow and Comb Hair Exercise', 'Shrug Exercise', 'Torso Twist Exercise', 
    'Reach Overhead and Touch Ear', 'Pendulum Exercise', 'Wall Climbing Exercise', 
    'Circle Drawing Exercise', 'Pulley Exercise', 'Wash Back Exercise'
]

async def mock_pose_estimation_model(frames: list) -> str:
    """
    模拟姿态识别模型。
    - 输入: 8帧图像 (这里我们只检查数量)
    - 输出: 一个随机的动作名称
    - 模拟耗时: 0.2秒
    """
    print(f"INFO: [模型] 正在识别 {len(frames)} 帧...")
    if len(frames) != 8:
        # 在真实场景中，这里应该抛出异常
        return "Unknown Action"
    
    await asyncio.sleep(0.2)  # 模拟模型推理延迟
    
    # 70%的概率识别为一个动作，30%的概率返回'Transitioning'来模拟动作转换期
    if random.random() < 0.7:
        action = random.choice(ACTION_LIST)
    else:
        action = 'Transitioning'
        
    print(f"INFO: [模型] 识别结果: {action}")
    return action

async def mock_qwen_vl_model(frames: list, action: str) -> dict:
    """
    模拟Qwen-VL多模态大模型进行评估。
    - 输入: 一个动作的所有帧图像和动作名称
    - 输出: 结构化的评估结果JSON
    - 模拟耗时: 1.5秒
    """
    print(f"INFO: [大模型] 正在评估动作 '{action}' (共 {len(frames)} 帧)...")
    await asyncio.sleep(1.5)  # 模拟大模型推理延迟

    # 创建一个模拟的评估报告
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

