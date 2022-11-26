python -B train.py --device_target GPU --device_id 0 --batch_size 6 --val_area Area_1 --scale --name randla_Area-1-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/randla_Area-1-gpu --val_area Area_1 --device_id 0 --device_target GPU
python -B train.py --device_target GPU --device_id 0 --batch_size 6 --val_area Area_2 --scale --name randla_Area-2-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/randla_Area-2-gpu --val_area Area_2 --device_id 0 --device_target GPU
python -B train.py --device_target GPU --device_id 0 --batch_size 6 --val_area Area_3 --scale --name randla_Area-3-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/randla_Area-3-gpu --val_area Area_3 --device_id 0 --device_target GPU
python -B train.py --device_target GPU --device_id 0 --batch_size 6 --val_area Area_4 --scale --name randla_Area-4-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/randla_Area-4-gpu --val_area Area_4 --device_id 0 --device_target GPU
python -B train.py --device_target GPU --device_id 0 --batch_size 6 --val_area Area_5 --scale --name randla_Area-5-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/randla_Area-5-gpu --val_area Area_5 --device_id 0 --device_target GPU
python -B train.py --device_target GPU --device_id 0 --batch_size 6 --val_area Area_6 --scale --name randla_Area-6-gpu --outputs_dir ./runs
python -B eval.py --model_path runs/randla_Area-6-gpu --val_area Area_6 --device_id 0 --device_target GPU
mkdir ./6_fold_result
cp -r ./runs/randla_Area-1-ascend/test_result/val_preds/*.ply
cp -r ./runs/randla_Area-2-ascend/test_result/val_preds/*.ply
cp -r ./runs/randla_Area-3-ascend/test_result/val_preds/*.ply
cp -r ./runs/randla_Area-4-ascend/test_result/val_preds/*.ply
cp -r ./runs/randla_Area-5-ascend/test_result/val_preds/*.ply
cp -r ./runs/randla_Area-6-ascend/test_result/val_preds/*.ply
python -B 6_fold_cv.py