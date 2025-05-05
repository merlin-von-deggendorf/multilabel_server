#!/bin/bash
# filepath: /root/serverupdater.sh

# Define variables
REPO_URL="https://github.com/merlin-von-deggendorf/microbify.git"
TARGET_DIR="/root/microbify/"

# Clone the repository
if [ -d "$TARGET_DIR" ]; then
  cd "$TARGET_DIR"
  git pull
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

# Recursively copy all data from sharedmodels to models
cp -r "$TARGET_DIR/sharedmodels/." "$TARGET_DIR/models/"

systemctl restart microbify.service