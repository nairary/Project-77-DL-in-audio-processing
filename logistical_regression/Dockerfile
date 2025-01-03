FROM python:3.10

WORKDIR /logistical_regression

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/

COPY . .

EXPOSE 8000

CMD ["python", "model_trainer/main.py"]