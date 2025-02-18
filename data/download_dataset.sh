#!/bin/bash

NUM_PROC=6

# Dataset split to download.
# Options: train, test, index.
SPLIT=$1

# Inclusive upper limit for file downloads. Should be set according to split:
# train --> 499.
# test --> 19.
# index --> 99.
N=$2

download_check_and_extract() {
  local i=$1
  images_file_name=images_$1.tar
  images_md5_file_name=md5.images_$1.txt
  images_tar_url=https://s3.amazonaws.com/google-landmark/$SPLIT/$images_file_name
  images_md5_url=https://s3.amazonaws.com/google-landmark/md5sum/$SPLIT/$images_md5_file_name
#   echo $images_tar_url
#   echo $images_md5_url
  echo "Downloading $images_file_name and its md5sum..."
#   curl -Os $images_tar_url > /dev/null
#   curl -Os $images_md5_url > /dev/null
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    images_md5="$(md5sum "$images_file_name")"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    images_md5="$(md5 -r "$images_file_name")"
  fi
  md5_1="$(cut -d' ' -f1<<<"$images_md5")"
  md5_2="$(cut -d' ' -f1<<<cat "$images_md5_file_name")"
  if [[ "$md5_1" != "" && "$md5_1" = "$md5_2" ]]; then
    tar -xf ./$images_file_name
    echo "$images_file_name extracted!"
  else
    echo "MD5 checksum for $images_file_name did not match checksum in $images_md5_file_name"
  fi
}

for i in $(seq 0 $NUM_PROC $N); do
  upper=$(expr $i + $NUM_PROC - 1)
  limit=$(($upper>$N?$N:$upper))
  for j in $(seq -f "%03g" $i $limit); do download_check_and_extract "$j" & done
  wait
done