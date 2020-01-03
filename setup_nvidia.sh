#!/bin/bash
#################################################################
## Script for installing Nvidia Drivers
#################################################################

DIR=$PWD
INSTDIR=$HOME

# NOTE: This requires GNU getopt.  On Mac OS X and FreeBSD, you have to install this
# separately; see below.
TEMP=`getopt -o unsk --long dir: -- "$@"`
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    --dir ) INSTDIR=$2; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

sudo wget -P $INSTDIR https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
sudo sh $INSTDIR/cuda_10.1.168_418.67_linux.run --override
