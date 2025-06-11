# Breast-COMA-Rehab

这是一个基于FastAPI和WebSocket的轻量级后端服务，用于实时接收和处理康复动作视频流。

## 安装依赖
请确保您已安装 Python 3.8+。然后运行以下命令安装所需库：

pip install -r requirements.txt

## 运行服务
在项目根目录下，使用uvicorn来启动服务：
```
uvicorn main:app --reload
```
服务将默认在 http://127.0.0.1:8000 启动。

## 如何测试
您需要一个WebSocket客户端来与此后端进行交互。

连接地址: ws://127.0.0.1:8000/ws/assessment

交互协议:

客户端 -> 服务端: 发送二进制消息，每条消息包含一张图片的字节流（例如，JPEG或PNG格式）。

服务端 -> 客户端: 发送JSON格式的文本消息。消息结构如下：
```
{
  "type": "info" | "evaluation",
  "data": "..." | { ... }
}
```
type: "info": 表示一个普通的文本通知消息。

type: "evaluation": 表示一个完整的动作评估报告。

## 前端实现建议
在前端，您可以使用`canvas`元素来捕获摄像头视频的每一帧，然后将其转换为`Blob`或`ArrayBuffer`，再通过WebSocket发送。
```
// 前端伪代码示例
const videoElement = document.getElementById('user-video');
const canvasElement = document.getElementById('capture-canvas');
const context = canvasElement.getContext('2d');
const ws = new WebSocket('ws://127.0.0.1:8000/ws/assessment');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('收到来自后端的消息:', message);
    if (message.type === 'evaluation') {
        // 在这里处理和展示评估报告
        displayReport(message.data);
    }
};

function sendFrame() {
    if (ws.readyState === WebSocket.OPEN) {
        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        canvasElement.toBlob((blob) => {
            ws.send(blob);
        }, 'image/jpeg', 0.8); // 发送JPEG格式，质量80%
    }
}

// 每秒发送15帧（大约66毫秒一帧）
setInterval(sendFrame, 66);
```