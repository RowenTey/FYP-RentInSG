# Use the official Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
# COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir streamlit pandas duckdb==0.9.2 plotly geopy requests numpy scikit-learn==1.4.2 catboost

# Copy the app files to the working directory
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]