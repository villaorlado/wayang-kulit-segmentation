#!/bin/sh

# Path to the CSV file
FILE="restore_ids.txt"

# Read the CSV file line by line
while IFS=',' read -r col1
do
  echo $col1
  break
  # aws s3api restore-object --bucket wayang-yt --key data/thumbnails/thumbnails_15secsPerFrame_320px240px/$col1.tar --restore-request '{"Days":5,"GlacierJobParameters":{"Tier":"Standard"}}'
done < "$FILE"

