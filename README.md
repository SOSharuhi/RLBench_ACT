

# RLBench_ACT: Running ALoha ACT and Diffusion Policy in the RLBench Framework

## Declaration

This repo is forked from the [Aloha ACT](https://github.com/tonyzhaozh/act), [RLBench](https://github.com/stepjam/RLBench), [Pyrep](https://github.com/stepjam/PyRep).

## Installation(Ubuntu20.04)

RLBench-ACT is built around ACT, RLBench, PyRep and CoppeliaSim v4.1.0. And we recommend [Conda](https://github.com/conda-forge/miniforge) as your python version manager!

1. Creating a python virtual environment RLBench_ACT

```bash
conda create -n rlbench_act python=3.8.10 # the version is strict
conda activate rlbench_act
```

2. Download [CoppeliaSim Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz). Add the following to your *~/.bashrc* file
    
```bash
export COPPELIASIM_ROOT=~/COPPELIASIM # you can change this path to where you want
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

```bash
# run and test CoppeliaSim
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
ln -s $COPPELIASIM_ROOT/libcoppeliaSim.so $COPPELIASIM_ROOT/libcoppeliaSim.so.1
bash $COPPELIASIM_ROOT/coppeliaSim.sh  
```

3. install this repository
    
```bash
# git the project
conda activate rlbench_act
git clone https://github.com/Boxjod/RLBench_ACT.git
cd RLBench_ACT

# install all requirements
conda activate rlbench_act
pip3 install -r requirements.txt
pip3 install -e ./PyRep # more information on https://github.com/stepjam/PyRep
pip3 install -e ./RLBench # more information on https://github.com/stepjam/RLBench
pip3 install -e ./act # more information on https://github.com/tonyzhaozh/act
pip3 install -e ./act/detr

# install pytorch-cuda
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

```

## Usage

1. test RLBench task builder. 
    
```bash
conda activate rlbench_act
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5 
#[remember don't save scene in Coppeliasim GUI ]
# if you occur error "Handle cam_head_mask does not exist.". go
```
Do not save the scence in Coppeliasim's GUI, either with *ctrl+s* or in the “Do you wish to save the changes?” window that pops up when you close it, you need to select *No* in all GUI screens. If you accidentally saved it in the GUI, run the following command:

```bash
cd rm task_design.ttt 
rm task_design.ttt 
cp task_design_back.ttt task_design.ttt
```

2. get demo for RPT in RLBench
    
```bash
python3 RLBench/tools/dataset_generator_hdf5.py \
--save_path Datasets \
--tasks sorting_program5 \
--variations 1 \
--episodes_per_task 50 \
--onscreen_render=False
```

3. visualize episode

```bash
python3 act/visualize_episodes.py --dataset_dir Datasets/sorting_program5/variation0 --episode_idx 0
```
4. train task and eval
    
```bash
# train
python3 act/imitate_episodes_rlbench.py \
--task_name sorting_program5 \
--ckpt_dir Trainings/sorting_program5 \
--policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 14000  --lr 1e-5 \
--seed 0

# infrence
python3 act/imitate_episodes_rlbench.py \
--task_name sorting_program5 \
--ckpt_dir Trainings/sorting_program5 \
--policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 14000  --lr 1e-5 --temporal_agg \
--seed 0 --eval --onscreen_render
```

The task is more difficult for the robot because there are 2 other colors of interferences at the same time, and the cube also comes with a certain angle of rotation of 45°, which can also lead to failure if the angles are not aligned. The success rate only goes up when the training epoch is around 14000.

## Task build

1. We recommend creating a new task from an already existing task. For example

```bash
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5
```
input the 'u' in the terminal window you can duplicate the task to a new name.

After your change, remember to save the task with the 's' in the terminal window. And you can test the task with "+" and "d".
But don't save this task scence in the CoppeliaSim GUI!!!

2. Edit the waypoints. And double-click on the waypoints and select Common Modification Extension string at the top of the pop-up window. The command list are bellow:

- ignore_collisions;
- open_gripper();
- close_gripper();
- steps(12); # You can set a fixed number of steps to reach this path point.
