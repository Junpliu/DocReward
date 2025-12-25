# need to replace <code_dir>, <mounted_code_dir>, <ckpt_dir>, <mounted_ckpt_dir> with actual paths before running
docker run \
  --privileged \
  --net=host \
  --ipc=host \
  --gpus=all \
  --runtime=nvidia \
  -v <code_dir>:<mounted_code_dir> \
  -v <ckpt_dir>:<mounted_ckpt_dir> \
  -w <mounted_code_dir> \
  -it hiyouga/llamafactory:0.9.3
  