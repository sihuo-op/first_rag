FROM python:3.13-slim

WORKDIR /app

# 创建并配置国内Debian镜像源
RUN echo "deb http://mirrors.ustc.edu.cn/debian/ trixie main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/debian/ trixie-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/debian/ trixie-backports main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/debian-security/ trixie-security main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY . .

RUN mkdir -p data/uploads data/sqlite

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
