# sinfo -O Nodehost,Gres:.30,GresUsed:.45

salloc -N 1 -n 16 --gres=gpu:NVIDIATITANV:1
# salloc -N 1 -n 16 --gres=gpu:TeslaM4024GB:1