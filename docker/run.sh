cd docker
export GID=$(id -g)
export GROUPNAME=$(echo $(id -Gn) | awk '{print $1}')
docker compose down
docker compose up -d --build
docker compose exec ultralytics_ws bash
cd ..
