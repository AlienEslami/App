FROM python:3.11-slim

RUN apt-get update && apt-get install -y glpk-utils libglpk-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install glpk

COPY . .

CMD ["python", "app.py"]
```

And make sure your `requirements.txt` does **not** contain `glpk` — it should only have:
```
flask
pyomo
pandas
openpyxl
