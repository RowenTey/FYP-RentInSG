docker compose down -v
sudo rm -rf logs/*
rm -f airflow.cfg
# docker volume rm scraper_data
# docker volume prune