#!/bin/sh

if [ -z "$1" ]; then
  echo "Usage: $0 <scraper-name>"
  exit 1
fi

SCRAPER=$1
DATE=$(date +"%Y-%m-%d")
echo "Running $SCRAPER.py for $DATE"
exec python pkg/scrapers/$SCRAPER.py > pkg/logs/scraper/${SCRAPER}_${DATE}.log 2>&1
