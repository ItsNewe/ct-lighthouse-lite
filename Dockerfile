# Builder stage
FROM python:3.13-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runner stage
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install only runtime dependencies for mysqlclient
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder 
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "main.py"]