#!/bin/bash

sudo apt-get update; sudo apt-get upgrade -y

echo "alias python=python3" >> $HOME/.profile
source $HOME/.profile

sudo apt-get install python python3-pip -y

pip3 install scikit-learn pytorch-pretrained-bert
