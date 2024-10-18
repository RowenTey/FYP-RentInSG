FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt ./requirements.txt

# Install the required packages
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the app files to the working directory
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Set the command to run the Streamlit app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]