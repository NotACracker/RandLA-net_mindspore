#!/usr/bin/env bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

for i in {1..6}
do
    python -B eval.py --model_path runs/randla_Area-$i-ascend --val_area Area_$i --device_id 0 --device_target Ascend --batch_size 32
done
mkdir ./6_fold_result
for i in {1..6}
do
    cp -r ./runs/randla_Area-$i-ascend/test_result/val_preds/*.ply ./6_fold_result
done
python -B 6_fold_cv.py