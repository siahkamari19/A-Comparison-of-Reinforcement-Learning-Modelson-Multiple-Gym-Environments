This is a SAC implementation for openai environments CartPole-v1, LunarLander-v2 and Acrobot-v1.

Packages Required:
tensorflow==1.14
gym
keras

Train a model:
python main.py --game {val} --mode train
val: 0 -> CartPole, 1 -> Acrobot, 2 -> LunarLander

Test a model:
python main.py --game {val} --mode test --path {model_path}
model_path: path of the .h5 model file.