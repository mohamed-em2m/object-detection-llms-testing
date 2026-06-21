export CUDA_VISIBLE_DEVICES=0,1  # Use only the first two GPUs
sudo apt-get update
sudo apt-get install libnvidia-compute-535
sudo apt-get install libnvidia-compute-535-server
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j 24 --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
install -m 755 llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
