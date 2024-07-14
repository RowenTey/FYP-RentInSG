#!/bin/sh

# Check if the first argument (scraper name) is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <scraper-name> [<debug-mode>]"
  exit 1
fi

# Set SCRAPER to the first argument
SCRAPER=$1

# Check if DEBUG_MODE environment variable is set
if [ "$DEBUG_MODE" = "true" ]; then
  DEBUG_FLAG="--debug"
fi

# Get the current date
DATE=$(date +"%Y-%m-%d")
echo "Running $SCRAPER.py for $DATE"

# Run the scraper with or without debug mode based on the DEBUG_FLAG
if [ -n "$DEBUG_FLAG" ]; then
  echo "Debug mode enabled"
  exec python pkg/scrapers/$SCRAPER.py -d
else
  exec python pkg/scrapers/$SCRAPER.py > pkg/logs/scraper/${SCRAPER}_${DATE}.log 2>&1
fi
