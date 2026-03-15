$env:RANK="0"
$env:LOCAL_RANK="0"
$env:WORLD_SIZE="1"
$env:MASTER_ADDR="localhost"
$env:MASTER_PORT="29500"

python ../../../wenet/bin/train.py --train_engine torch_ddp --device cpu --config conf/train_conformer.yaml --data_type raw --train_data data/train/data.list --cv_data data/dev/data.list --model_dir exp/pogo_test --num_workers 0 --ddp.dist_backend gloo
