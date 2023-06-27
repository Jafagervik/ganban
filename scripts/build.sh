#!/usr/bin/bash  
 

echo "Now training net.."
python train.py --model Net --batch_size 32 --lr 1e-4 --epochs 30
