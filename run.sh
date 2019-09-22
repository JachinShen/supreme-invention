#!/bin/sh
python -u train.py --seed 233 --enemy hand --load_model --load_model_path "ICRA.model" --save_model_path "ICRA_save.model" --epoch 1000 --update_step 10

python -u train.py --seed 233 --enemy hand --load_model --load_model_path "ICRA.model" --epoch 50