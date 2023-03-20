
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=iscilab/segmentation:cuda-20-04
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        segmentation \
        $BASH_OPTION
else
    docker start -i segmentation
fi
