# mobilenet_compression
Compress MobileNet V1 and V2 using tools from NNI

# preparation
```
pip install -r requirements.txt
chmod u+x prepare_data.sh
./prepare_data.sh
```

# pretraining
```
python pretrain.py
```

# experiment with pruning
```
python pruning_experiments.py [arguments]
```
