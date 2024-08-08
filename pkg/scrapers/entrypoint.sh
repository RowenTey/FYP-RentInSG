#!/bin/sh

# Check if the first argument (scraper name) is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <scraper-name>"
  exit 1
fi

# Set SCRAPER to the first argument
SCRAPER=$1

# Get the current date
DATE=$(date +"%Y-%m-%d")
echo "Running $SCRAPER.py for $DATE"

# Run the scraper in different modes depending on the arguments
if [ "$DEBUG_MODE" = "true" ]; then
  echo "Debug mode enabled"
  exec python -u pkg/scrapers/$SCRAPER.py -d
elif [ "$LOG_OUTPUT" = "false" ]; then
  echo "Log output disabled"
  exec python -u pkg/scrapers/$SCRAPER.py
else
  exec python pkg/scrapers/$SCRAPER.py > pkg/logs/scraper/${SCRAPER}_${DATE}.log 2>&1
fi

