# ICRA 2019 Simulator

## Requirements

- Python3
- Pytorch
- OpenAI gym
- Box2D
- swig

## Installation

### Anaconda

```
conda create -f Gym.yaml
conda activate Gym
```

### Pip

```
sudo apt-get install swig # or install from source
pip3 install gym box2d box2d-kengz
pip3 install pytorch
```

## Try the simulator

```
python3 ICRAField.py
```


## Train

```
python3 train.py
```

## Test

```
python3 test.py
```

## Screenshot
![](imgs/screenshot.png)

## TODO

1. ~~ICRA Map Construction~~
2. ~~Bullet Simulation~~
3. ~~Damage and blood calculation~~
4. ~~Simple strategy implemention~~
5. ~~Path planning~~
6. ~~Moving behaviour with mecanum wheels~~
7. Run multiple simulators in parallel
8. Implement complete model
9. Implement the same path planning in RoboRTS
10. Rearrange the use of dictionary and array
11. Measure hysicial parameters
11. Record video in a headless server
