# Breast-COSMA

这是一个基于FastAPI和WebSocket的轻量级后端服务，用于实时接收和处理康复动作视频流。

## 安装依赖
请确保您已安装 Python 3.8+。然后运行以下命令安装所需库（mmcv需要修改源码使得版本兼容）：

```
pip install -r requirements.txt

bash init.sh
```


## 运行服务
在项目根目录下，使用uvicorn来启动服务：
```
nohup gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 run:app --reload > gunicorn.log 2>&1 &
```
服务将默认在 http://127.0.0.1:5000 启动。
