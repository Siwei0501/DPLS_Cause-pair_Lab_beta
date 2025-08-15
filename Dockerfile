

FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
 && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8501
CMD ["streamlit", "run", "app/DPLS_Lab.py", "--server.port=8501", "--server.address=0.0.0.0"]
