@echo off
set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1
set MASTER_ADDR=localhost
set MASTER_PORT=29500

.venv\Scripts\python.exe ..\..\..\wenet\bin\train.py --train_engine torch_ddp --device cpu --config conf\train_conformer.yaml --data_type raw --train_data data\train\data.list --cv_data data\dev\data.list --model_dir exp\pogo_test --num_workers 0 --ddp.dist_backend gloo
