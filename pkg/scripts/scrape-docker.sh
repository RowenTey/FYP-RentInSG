#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <telegram-token> <chat-id>"
  exit 1
fi

# Function to stop and remove a container with a specific name
cleanup_specific_container() {
  container_name=$1
  container_id=$(docker ps -aqf "name=${container_name}")

  if [ -n "$container_id" ]; then
    echo "Stopping container: $container_name..."
    docker stop $container_name

    echo "Removing container: $container_name..."
    docker rm $container_name
  else
    echo "No container found with the name: $container_name"
  fi
}

# Function to run a specific Docker container
run_container() {
  container_name=$1
  
  docker run -d \
    --name $container_name \
    -e TELEGRAM_TOKEN=$1 \
    -e TELEGRAM_CHAT_ID=$2 \
    -v /home/ubuntu/FYP-RentInSG/pkg/logs/scraper:/app/pkg/logs/scraper \
    -v /home/ubuntu/FYP-RentInSG/pkg/rental_prices/propnex:/app/pkg/rental_prices/propnex \
    rowentey/fyp-rent-in-sg:propnex-scraper-latest
}

# Main script execution
container_name="propnex-scraper" 

echo "Cleaning up container: $container_name..."
cleanup_specific_container $container_name

echo "Running the specific Docker container..."
run_container $container_name

echo "Done"
