#!/bin/bash

# NRP DEVELOPER, VERY IMPORTANT: increment the VERSION on every commit, since it will force an update on users' side

VERSION=1.14

auto_update_script() {
  set +e
  scriptname=$1
  echo -e "${BLUE}Checking for script updates${NC}"
  wget --no-check-certificate https://neurorobotics-files.net/index.php/s/HQJzj8fywKN8oxZ/download -O /tmp/$scriptname 2>/dev/null
  if [ -f /tmp/$scriptname -a -n ""`grep VERSION= /tmp/$scriptname | grep -v grep` ]; then
    newversion=`grep VERSION= /tmp/$scriptname | grep -v grep | sed -n "s/[^=]*=\([0-9]*.*\)/\1/p"`
    version_check $VERSION $newversion
    check=$?
    if [ "$check" -eq 9 ]; then
      echo -e "${YELLOW}A newer recent of this script has been found. Update (recommended)? [y/n] ${NC}"
      read yn
      case $yn in
        [Yy]* ) mv -f /tmp/$scriptname ./$scriptname
                chmod 755 ./$scriptname
                echo -e "${GREEN}$scriptname has been updated and will exit now. Just re-run it.${NC}"
                exit
                ;;
        * ) ;;
      esac
    else
      echo -e "${GREEN}$scriptname is up-to-date.${NC}"
    fi
  fi
  set -e
}

restart() {
  container=$1
  echo -e "${BLUE}Restarting $container${NC}"
  $DOCKER_CMD restart $container && $DOCKER_CMD exec $container bash -c "sudo /etc/init.d/supervisor start"
  echo -e "${GREEN}$container container has now been restarted.${NC}"
}

stop() {
  container=$1
  echo -e "${BLUE}Stopping $container${NC}"
  $DOCKER_CMD stop $container
  echo -e "${GREEN}$container container has now been stopped.${NC}"
}

delete() {
  container=$1
  while true; do
    echo -e "${YELLOW}$container container will be deleted. You will lose any data you may have changed inside it (but not your experiments and models). Can we proceed? [y/n] ${NC}"
    read yn
    case $yn in
      [Yy]* )
              echo -e "${BLUE}Deleting $container container${NC}"
              $DOCKER_CMD rm $container
              echo -e "${GREEN}Successfully deleted old $container container.${NC}"
              return 0;;
      [Nn]* ) return 1;;
      * ) echo "Please answer yes or no.";;
    esac
  done
}

restore() {
  container=$1
  set +e
  stop $1
  delete $1 && start $1 || restart $1
  set -e
}

start(){
  container=$1
  eval "port=\${${container}_port}"
  eval "image=\${${container}_image}"
  check_port $port
  ipvar=$container"_ip"
  echo -e "${BLUE}Starting $container container on port $port using image $image${NC}"
  iparg=`eval $is_mac || echo --ip=${!ipvar}`
  $DOCKER_CMD run -itd \
    -p $port:$port \
    --net=nrpnet \
    $iparg \
    -v nrp_user_data:/home/bbpnrsoa/.opt/nrpStorage \
    -v nrp_models:/home/bbpnrsoa/nrp/src/Models \
    -v nrp_experiments:/home/bbpnrsoa/nrp/src/Experiments \
    --name $container $image
  eval setup_$container
  echo -e "${GREEN}$container container is now up and running.${NC}"
}

pull_images(){
  echo -e "${BLUE}Pulling frontend image, this may take a while..${NC}"
  $DOCKER_CMD pull hbpneurorobotics/nrp_frontend:dev
  echo -e "${GREEN}Successfully downloaded frontend image.${NC}"
  echo -e "${BLUE}Pulling nrp image, this may take even longer..${NC}"
  $DOCKER_CMD pull hbpneurorobotics/nrp:dev
  echo -e "${GREEN}Successfully downloaded nrp image.${NC}"
  set +e
  $DOCKER_CMD network create -d bridge --subnet $subnet.0.0/16 --gateway $subnet.0.1 nrpnet
  set -e
  $DOCKER_CMD volume create nrp_models
  $DOCKER_CMD volume create nrp_experiments
  $DOCKER_CMD volume create nrp_user_data
  for container in nrp frontend
  do
    if [[ $($DOCKER_CMD ps -a | grep -w $container$) ]]
    then
      echo -e "A $container container is already running."
      restore $container
    else
      start $container
    fi
  done
  echo -e "${BLUE}Removing old unused images${NC}"
  $DOCKER_CMD system prune
  echo ""
  echo -e "${GREEN}
Congratulations! The NRP platform is now installed on your computer.
${NC}
You can check everything works by going to ${PURPLE}http://localhost:9000/#/esv-private ${NC}or if you used the --ip option: ${PURPLE}http://$external_frontend_ip:9000/#/esv-private ${NC}by using your browser and signing in with the following credentials:

username : nrpuser
password : password

If you need any help please use our forum: ${PURPLE}https://forum.humanbrainproject.eu/c/neurorobotics${NC}"
}

setup_nrp(){
  echo -e "${BLUE}Setting up nrp container${NC}"
  $DOCKER_CMD exec nrp bash -c 'echo "127.0.0.1 $(uname -n)" | sudo tee --append /etc/hosts'
  set +e
  echo -e "${BLUE}Cloning template models, this may take a while${NC}"
  $DOCKER_CMD exec nrp bash -c '{ cd /home/bbpnrsoa/nrp/src/Models && sudo git pull; } || { cd /home/bbpnrsoa/nrp/src && sudo find Models/ -not -name "Models" -delete && sudo git clone --progress --branch=master https://bitbucket.org/hbpneurorobotics/Models.git Models/; }'
  echo -e "${BLUE}Cloning template experiments, this may take a while${NC}"
  $DOCKER_CMD exec nrp bash -c '{ cd /home/bbpnrsoa/nrp/src/Experiments && sudo git pull; } || { cd /home/bbpnrsoa/nrp/src && sudo find Experiments/ -not -name "Experiments" -delete && sudo git clone --progress --branch=master https://bitbucket.org/hbpneurorobotics/Experiments.git Experiments/; }'
  $DOCKER_CMD exec nrp bash -c 'sudo chown -R bbpnrsoa:bbp-ext /home/bbpnrsoa/nrp/src/Experiments && sudo chown -R bbpnrsoa:bbp-ext /home/bbpnrsoa/nrp/src/Models'
  set -e
  echo -e "${BLUE}Setting rendering mode to CPU${NC}"
  $DOCKER_CMD exec nrp bash -c '/home/bbpnrsoa/nrp/src/user-scripts/rendering_mode cpu'
  echo -e "${BLUE}Generating low resolution textures${NC}"
  $DOCKER_CMD exec nrp bash -c 'python /home/bbpnrsoa/nrp/src/user-scripts/generatelowrespbr.py'
  $DOCKER_CMD exec nrp bash -c 'export NRP_MODELS_DIRECTORY=$HBP/Models && /home/bbpnrsoa/nrp/src/Models/create-symlinks.sh' 2>&1 | grep -v "HBP-NRP"
  $DOCKER_CMD exec nrp bash -c "/bin/sed -e 's/localhost:9000/"$external_frontend_ip":9000/' -i /home/bbpnrsoa/nrp/src/ExDBackend/hbp_nrp_backend/hbp_nrp_backend/config.py"
  $DOCKER_CMD exec nrp bash -c "/bin/sed -e 's/localhost:9000/"$external_frontend_ip":9000/' -i /home/bbpnrsoa/nrp/src/VirtualCoach/hbp_nrp_virtual_coach/hbp_nrp_virtual_coach/config.json"
  $DOCKER_CMD exec nrp bash -c 'cd $HOME/nrp/src && source $HOME/.opt/platform_venv/bin/activate && pyxbgen -u Experiments/bibi_configuration.xsd -m bibi_api_gen && pyxbgen -u Experiments/ExDConfFile.xsd -m exp_conf_api_gen && pyxbgen -u Models/environment_model_configuration.xsd -m environment_conf_api_gen && pyxbgen -u Models/robot_model_configuration.xsd -m robot_conf_api_gen && deactivate' 2>&1 | grep -v "WARNING"
  $DOCKER_CMD exec nrp bash -c 'gen_file_path=$HBP/ExDBackend/hbp_nrp_commons/hbp_nrp_commons/generated && filepaths=$HOME/nrp/src && sudo cp $filepaths/bibi_api_gen.py $gen_file_path &&  sudo cp $filepaths/exp_conf_api_gen.py $gen_file_path && sudo cp $filepaths/_sc.py $gen_file_path && sudo cp $filepaths/robot_conf_api_gen.py $gen_file_path && sudo cp $filepaths/environment_conf_api_gen.py $gen_file_path'
  $DOCKER_CMD exec nrp bash -c "sudo /etc/init.d/supervisor start"
  echo -e "${GREEN}Finished setting up nrp container.${NC}"
}

setup_frontend() {
  $DOCKER_CMD exec frontend bash -c "/bin/sed -e 's/localhost/"$external_frontend_ip"/' -i /home/bbpnrsoa/nrp/src/ExDFrontend/dist/config.json"
  $DOCKER_CMD exec frontend bash -c "/bin/sed -e \"s=localhost="$external_nrp_ip"=\" -i /home/bbpnrsoa/nrp/src/nrpBackendProxy/config.json"
  # Remove exit on fail, as if the user exists already we dont care.
  set +e
  $DOCKER_CMD exec frontend bash -c "source /home/bbpnrsoa/nrp/src/user-scripts/nrp_variables 2> /dev/null && /home/bbpnrsoa/nrp/src/user-scripts/add_new_database_storage_user -u nrpuser -p password -s > /dev/null 2>&1"
  set -e
  $DOCKER_CMD exec frontend bash -c "sudo /etc/init.d/supervisor start"
  echo -e "${GREEN}Finished setting up frontend container.${NC}"
}

uninstall(){
  while true; do
    echo -e "${YELLOW}Are you sure you would like to remove the NRP docker images? [y/n] ${NC}"
    read yn
    case $yn in
      [Yy]* ) break;;
      [Nn]* ) exit;;
      * ) echo "Please answer yes or no.";;
    esac
  done
  # Dont fail on errors
  set +e
  echo -e "${BLUE}Removing NRP docker images. This may take a while.${NC}"
  $DOCKER_CMD stop nrp
  $DOCKER_CMD stop frontend
  $DOCKER_CMD rm nrp
  $DOCKER_CMD rm frontend
  $DOCKER_CMD network rm nrpnet
  $DOCKER_CMD rmi $($DOCKER_CMD images | grep -w hbpneurorobotics/nrp | awk '{print $3}')
  $DOCKER_CMD rmi $($DOCKER_CMD images | grep -w hbpneurorobotics/nrp_frontend | awk '{print $3}')
  $DOCKER_CMD volume rm nrp_models
  $DOCKER_CMD volume rm nrp_experiments
  echo -e "${GREEN}NRP Docker images have been successfully removed.${NC}"
  set -e
  while true; do
    echo -e "${YELLOW}Would you also like to remove your personal experiments and models? [y/n] ${NC}"
    read yn
    case $yn in
      [Yy]* ) break;;
      [Nn]* ) exit;;
      * ) echo "Please answer yes or no.";;
    esac
  done
  $DOCKER_CMD volume rm nrp_user_data
  echo -e "${BLUE}Removing NRP user data${NC}"
  echo -e "${GREEN}All traces of the NRP images and user data have been sucessfully removed from your system.${NC}"
}

connect(){
  container=$1
  echo -e "${BLUE}Opening new terminal into $container container${NC}"
  thecmd="bash -c \"echo -e \\\"${RED}You are inside the $container container. Advanced users only.\nCTRL+D to exit.\nIf you mess up everything, you can restore this container\nwith the reset command of the install script.\n${NC}\\\"; $DOCKER_CMD exec -it $container bash\""
  if [ -z ""`which gnome-terminal` ]
  then
    echo -e "${GREEN}No gnome-terminal installed. Defaulting to bash.${NC}"
    bash -c "$thecmd"
  else
    gnome-terminal -e "$thecmd" &
  fi
}

check_port(){
  port=$1
  echo -e "${BLUE}Checking ports${NC}"
  set +e
  is_port=`netstat -tuplen 2>/dev/null | grep $port`
  if [ -n "$is_port" ]
  then
    echo -e "${RED}[ERROR] The port $port is in currently in use. If you would like to install the NRP please find the process using this port and stop it:${NC}"
    echo -e "$is_port"
    exit
  fi
  echo -e "${GREEN}Port $port is available.${NC}"
  set -e

}

version_check() {

   [ -z "$1" -o -z "$2" ] && return 9
   [ "$1" == "$2" ] && return 10

   ver1front=`echo $1 | cut -d "." -f -1`
   ver1back=`echo $1 | cut -d "." -f 2-`

   ver2front=`echo $2 | cut -d "." -f -1`
   ver2back=`echo $2 | cut -d "." -f 2-`

   if [ "$ver1front" != "$1" ] || [ "$ver2front" != "$2" ]; then
       [ "$ver1front" -gt "$ver2front" ] && return 11
       [ "$ver1front" -lt "$ver2front" ] && return 9

       [ "$ver1front" == "$1" ] || [ -z "$ver1back" ] && ver1back=0
       [ "$ver2front" == "$2" ] || [ -z "$ver2back" ] && ver2back=0
       version_check "$ver1back" "$ver2back"
       return $?
   else
           [ "$1" -gt "$2" ] && return 11 || return 9
   fi
}

#Fail on errors
set -e

is_mac=false
if grep -qE "(Microsoft|WSL)" /proc/version; then exe=".exe"; 
elif uname -a | grep -q "Darwin"
then
   is_mac=true;
fi
nrp_port="8080"
nrp_image="hbpneurorobotics/nrp:dev"
frontend_port="9000"
frontend_image="hbpneurorobotics/nrp_frontend:dev"
subnet="172.19"
if $is_mac
then
   frontend_ip="host.docker.internal"
   nrp_ip="host.docker.internal"
else
  frontend_ip="172.19.0.2"
  nrp_ip="172.19.0.3"
fi
external_frontend_ip=$frontend_ip
external_nrp_ip=$nrp_ip
DOCKER_CMD="docker"$exe
CMD=""
nrp_proxy_ip="http://148.187.97.48"
#Colours
RED="\033[01;31m"
GREEN="\033[01;32m"
PURPLE="\033[01;35m"
BLUE="\033[01;34m"
YELLOW="\033[01;33m"
NC="\033[00m"

usage="
Usage: $(basename "$0") COMMAND

A script to install and start the Neurorobotics Platform using docker images.

Options:
    -h                   Show this help text
    -s/--sudo            Use docker with sudo"
if ! $is_mac
then
usage="$usage
    -i/--ip <ip_address> The IP address of the machine the images are installed on.
                         Use this option when you would like to access the NRP outside the machine its installed on."
fi
usage="$usage
Commands:
    restart_backend   Restart the backend container
    restart_frontend  Restart the frontend container
    restart           Restart backend and frontend containers
    update            Update the NRP
    install           Install the NRP
    uninstall         Uninstall the NRP
    stop              Stops the nrp containers
    start             Starts nrp containers which have previously been stopped
    reset_backend     Restores the backend container
    reset_frontend    Restores the frontend container
    reset             Restores the backend and frontend containers
    connect_frontend  Connect to the frontend container (Opens in a new terminal)
    connect_backend   Connect to the backend container (Opens in a new terminal)

${BLUE}Please note:${NC}
This script requires that the package 'docker' is already installed
At least 15GB of disk space will be needed in order to install the NRP images${NC}
"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
      echo -e "$usage"
      exit
    ;;
    -s|--sudo)
      DOCKER_CMD="sudo docker"
      shift
    ;;
    -i|--ip)
      external_frontend_ip="$2"
      external_nrp_ip=$external_frontend_ip
      shift
      shift
    ;;

    restart_backend)
       CMD="restart nrp"
       shift
     ;;
    restart_frontend)
       CMD="restart frontend"
       shift
     ;;
    restart)
       CMD="restart nrp && restart frontend"
       shift
     ;;
    start)
       CMD="restart nrp && restart frontend"
       shift
     ;;
    update)
      set +e && curl -X POST ${nrp_proxy_ip}/proxy/activity_log/update --max-time 10; set -e # logs each update event via the NRP proxy server
      CMD="pull_images"
      shift
    ;;
    install)
      set +e && curl -X POST ${nrp_proxy_ip}/proxy/activity_log/install --max-time 10; set -e # logs each install event via the NRP proxy server
      CMD="pull_images"
      shift
    ;;
    uninstall)
      CMD="uninstall"
      shift
    ;;
    stop)
      CMD="stop nrp && stop frontend"
      shift
    ;;
    reset_backend)
      CMD="restore nrp"
      shift
    ;;
    reset_frontend)
      CMD="restore frontend"
      shift
    ;;
    reset)
       CMD="restore nrp && restore frontend"
       shift
     ;;
    connect_backend)
      CMD="connect nrp"
      shift
    ;;
    connect_frontend)
      CMD="connect frontend"
      shift
    ;;

    *)
     echo "Unknown option \"$key\""
     echo -e "$usage"
     exit
     ;;
esac
done

if [ -z "$CMD" ]
then
  echo -e "${RED}[ERROR] Please provide a command to execute${NC}"
  echo ""
  echo -e "$usage"
  exit
fi

auto_update_script `basename $0`
eval $CMD
# Reset terminal colour back to normal
echo -e "${NC}"
