python -B train.py --device_target GPU --device_id 0 --batch_size 6 --val_area Area_5 --scale --name randla_Area-5-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/randla_Area-5-gpu --val_area Area_5 --device_id 0 --device_target GPU --batch_size 20