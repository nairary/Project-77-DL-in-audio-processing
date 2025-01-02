FROM python:3.10

WORKDIR /logistical_regression

COPY data/ /app/data/

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "model_trainer/main.py"]