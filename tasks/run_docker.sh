NBA_PREDICTIONS_DIR=/Users/asanzgiri/nba_predictions

docker run -ti -v ${NBA_PREDICTIONS_DIR}:/home/jovyan/work  -e USER=${USER} -p 8888:8888 --ipc=host \
                  nba_elo_docker_image:0.0 jupyter notebook \
                  --no-browser --ip=0.0.0.0 --NotebookApp.token= --allow-root
