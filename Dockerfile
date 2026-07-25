 FROM python:3.12-slim

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up a dedicated non-root user to perfectly align with Hugging Face policies
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# FIXED: Explicitly map the container root directory to Python's structural paths
ENV PYTHONPATH=/home/user/app

# Copy requirements and install packages as the safe non-root user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user -r requirements.txt

# Transfer ownership of all source code files to the non-root user

# Explicitly transfer ownership of all source code files to the non-root user
COPY --chown=user . .

# Expose Streamlit (8501) and FastAPI (8000) ports
EXPOSE 8501
EXPOSE 8000

# Default command launches the Streamlit UI dashboard
CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.headless=true"]