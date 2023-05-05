
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=iscilab/uoais_ros2:cuda-20-04
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        uoais_ros2 \
        $BASH_OPTION
else
    docker start -i uoais_ros2
fi
