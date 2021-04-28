# PPO implementation on OpenAI Environments

Before executing run: pip install .

How to run:

python PPO/main.py --game {val}
val: Name of Env


Test Policy:

python PPO/test.py --policy_path {Policy_Path} \ 
                     --env_name {Name Of Env} --render {True|False}

Sample policy path in logs folder.
