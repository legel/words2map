#!/bin/bash
OS1="$(uname -m)"
OS2="$(uname -s)"

if [ $OS1 = "x86_64" ]; then
  echo "64 bit system identified"
elif [ $OS1 = "i686" ]; then
  echo "32 bit system identified"
fi

if [ $OS2 = "Linux" ]; then
  echo "Linux system identified"
elif [ $OS2 = "Darwin" ]; then
  echo "OSX system identified"
fi