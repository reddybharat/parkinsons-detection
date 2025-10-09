# Step 1: Use a small but capable base image
FROM python:3.12-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Install system dependencies for OpenCV and image processing
# - libgl1: for OpenCV GUI/image ops
# - libglib2.0-0: for image decoding
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements first (for layer caching)
COPY requirements.txt .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy your source code
COPY src/ ./src
COPY models/ ./models

# Step 7: Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Step 8: Run both backend (FastAPI) and frontend (Streamlit)
CMD ["bash", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0"]