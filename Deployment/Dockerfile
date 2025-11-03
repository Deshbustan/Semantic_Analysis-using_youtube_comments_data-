FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy and install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code and the mlruns folder
COPY . .

# 5. Expose the port the app will run on
EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app:app"]