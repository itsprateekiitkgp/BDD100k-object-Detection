# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency list and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your analysis script and any other necessary files
COPY . .

# By default, show help if no arguments are passed
ENTRYPOINT ["python", "analysis.py"]
CMD ["--help"]
