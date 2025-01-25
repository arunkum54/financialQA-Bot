# Use official Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create and switch to non-root user for security
RUN useradd -m appuser
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies globally
RUN pip install --no-cache-dir -r requirements.txt


# Explicitly install Streamlit globally
RUN pip install --no-cache-dir streamlit==1.33.0

# Add Streamlit to PATH (optional, but good practice)
ENV PATH="/home/appuser/.local/bin:${PATH}"
# Copy application code
COPY --chown=appuser:appuser app.py backend.py .

# Expose port and run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]