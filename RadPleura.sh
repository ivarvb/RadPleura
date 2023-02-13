#!/bin/bash
PW="./python/python38radpleura/"
APP="./sourcecode/src/vx/radpleura"

install () {
    #sudo apt-get install python3-venv -y
    rm -r $PW
    mkdir $PW
    python3 -m venv $PW
    source $PW"bin/activate"
    pip3 install wheel
    pip3 install --upgrade pip
    pip3 install --upgrade setuptools    
    
    pip3 install -r $APP"/xrequeriments.txt"
    compile
}
compile () {
    #ddddd
    pwd
}
execute(){
    source $PW"bin/activate"
    
    cd ./sourcecode/src/
    python3 RadPleura.py
}
polygon(){
    source $PW"bin/activate"
    
    cd ./sourcecode/src/vx/radpleura
    python3 Media.py
}
roi(){
    source $PW"bin/activate"
    
    cd ./sourcecode/src/vx/radpleura
    python3 ROI.py
}
regions(){
    source $PW"bin/activate"
    
    cd ./sourcecode/src/vx/radpleura/px/image
    bash Make.sh
}
features(){
    source $PW"bin/activate"
    
    cd ./sourcecode/src/vx/radpleura
    python3 Features.py
}
classification(){
    source $PW"bin/activate"
    
    cd ./sourcecode/src/vx/radpleura
    python3 Classification.py
}

roidl(){
    source $PW"bin/activate"
    
    cd ./sourcecode/src/vx/radpleura
    python3 DL.py
}


args=("$@")
T1=${args[0]}
if [ "$T1" = "install" ]; then
    install
elif [ "$T1" = "compile" ]; then
    compile
elif [ "$T1" = "polygon" ]; then
    polygon
elif [ "$T1" = "roi" ]; then
    roi
elif [ "$T1" = "regions" ]; then
    regions
elif [ "$T1" = "features" ]; then
    features
elif [ "$T1" = "classification" ]; then
    classification
elif [ "$T1" = "roidl" ]; then
    roidl
else
    execute
fi