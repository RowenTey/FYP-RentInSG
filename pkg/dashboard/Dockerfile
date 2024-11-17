# Use the official Python base image
# FROM python:3.11-slim
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./requirements.txt

# Install the required packages
# RUN pip install --upgrade pip && pip install --no-cache-dir streamlit pandas duckdb==0.10.2 plotly geopy requests numpy pygwalker==0.4.9.4 streamlit-elements shap==0.46.0 matplotlib==3.9.2
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files to the working directory
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]