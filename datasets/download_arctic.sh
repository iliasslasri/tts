#!/bin/bash
# chmod +x download_arctic.sh
# ./download_arctic.sh

URL="http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_awb_arctic-0.90-release.tar.bz2"

FILENAME=$(basename $URL)

echo "Downloading $FILENAME..."
wget $URL -O $FILENAME

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Download failed!"
    exit 1
fi

# Extract the tar.bz2 file
echo "Extracting $FILENAME..."
tar -xjf $FILENAME

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Extraction failed!"
    exit 1
fi

echo "Done!"
