#!/bin/bash

if [ ! -e glove-twitter ]; then
  if hash wget 2>/dev/null; then
    wget http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
  else
    curl -O http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
  fi
  mkdir glove-twitter
  unzip glove.twitter.27B.zip -d glove-twitter
  rm glove.twitter.27B.zip
fi