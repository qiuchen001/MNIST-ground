# 1. 选择一个包含 PyTorch 和 CUDA 的基础镜像
#    (您可以根据您集群的 CUDA 版本选择更具体的标签)
#FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# 2. 设置工作目录
WORKDIR /app

# 3. 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制您的训练脚本
COPY train.py .

# 5. 设置默认的执行命令
#    当容器启动时，它将运行 "python train.py"
#    所有参数将在 VolcanoJob YAML 中提供
ENTRYPOINT ["python", "train.py"]
