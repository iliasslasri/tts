#!/bin/bash
# chmod +x download_arctic.sh
# ./download_arctic.sh

URL="http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_awb_arctic-0.90-release.tar.bz2"
FILENAME=$(basename "$URL")

echo "Downloading $FILENAME..."
if ! wget "$URL" -O "$FILENAME"; then
    echo "Download failed!"
    exit 1
fi

echo "Extracting $FILENAME..."
if ! tar -xjf "$FILENAME"; then
    echo "Extraction failed!"
    exit 1
fi

echo "Done!"
