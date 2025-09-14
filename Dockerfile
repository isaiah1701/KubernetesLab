FROM python:3.12-alpine

WORKDIR /defaultApplication

COPY . /defaultApplication

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "defaultApplication.app:app", "--host", "0.0.0.0", "--port", "5000"]
