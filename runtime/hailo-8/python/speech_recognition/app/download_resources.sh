#!/usr/bin/bash

#!/bin/bash

# Check if the "hefs" directory exists and delete it if it does
if [ -d "hefs" ]; then
  echo "Deleting existing 'hefs' directory..."
  rm -rf hefs
fi

echo "Creating new 'hefs/h8/tiny' directory..."
mkdir -p hefs/h8/tiny

# Download the files using wget
echo "Downloading tiny-whisper-decoder for Hailo-8..."
wget -P hefs/h8/tiny "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/tiny-whisper-decoder-fixed-sequence-matmul-split.hef"

echo "Downloading tiny-whisper-encoder for Hailo-8..."
wget -P hefs/h8/tiny "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/tiny-whisper-encoder-10s_15dB.hef"


echo "Creating new 'hefs/h8/tiny' directory..."
mkdir -p hefs/h8l/tiny
echo "Downloading tiny-whisper-decoder for Hailo-8L..."
wget -P hefs/h8l/tiny "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef"

echo "Downloading tiny-whisper-encoder for Hailo-8L..."
wget -P hefs/h8l/tiny "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/tiny-whisper-encoder-10s_15dB_h8l.hef"


if [ -d "decoder_assets" ]; then
  echo "Deleting existing 'decoder_assets' directory..."
  rm -rf decoder_assets
fi


echo "Creating new 'decoder_assets/tiny' directory..."
mkdir -P decoder_assets/tiny
mkdir -P decoder_assets/tiny/decoder_tokenization
echo "Downloading decoder assets..."
wget -P decoder_assets/tiny/decoder_tokenization "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/onnx_add_input_tiny.npy"
wget -P decoder_assets/tiny/decoder_tokenization "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets/tiny/decoder_tokenization/token_embedding_weight_tiny.npy"


echo "Download complete."



