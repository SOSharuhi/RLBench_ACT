import os
import numpy as np
from sim_env_rlbench import make_sim_env
from rlbench.demo import Demo
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import math

from joyconrobotics import JoyconRobotics
import threading
import argparse
import h5py

VARIATIONS_FOLDER = 'variation%d'

def recoding_process(scene):
    collect_rate = 30
    print("start the recoding porcess")
    demo = []
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        # 记录数据
        demo.append(scene.get_observation())
        print(f'recoding_process working on {len(demo)}')
        time.sleep(1/collect_rate)
    return demo
        
class MyThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return_value = None

    def run(self):
        # 这里的 self._target 就是传进去的 doit
        if self._target:
            self._return_value = self._target(*self._args, **self._kwargs)

    def get_result(self):
        return self._return_value

def get_demo_joycon(joyconrobotics, env, right_zero_pos):
    # 另开一个线程记录数据
    recoder = MyThread(target=recoding_process, args=(env._scene,))
    recoder.start()
    
    path = []
    demo = []
    t = 0
    ts_obs = env._scene.get_observation()
    while(True):
        pose, gripper, control_button = joyconrobotics.get_control()
        right_target_pos = pose[0:3]
        orientation_rad = pose[3:6]

        r4 = R.from_euler('xyz', orientation_rad, degrees=False)
        orientation_output = r4.as_quat()
        right_target_quat = orientation_output
        
        frame_start_time = time.time()
        new_path = env._robot.arm.get_linear_path(position=right_target_pos, quaternion=right_target_quat, steps=5, relative_to=env._robot.arm, ignore_collisions=True)
        # read rate = 870.1875518672199 Hz
        
        if new_path != None: # if IK_success
            path.append(new_path)
            
            path[t].visualize() 
            done = False 
            while done != 1: 
                done = path[t].step() 
                env._scene.step() 

            env._action_mode.gripper_action_mode.action(env._scene, np.array((gripper,)))
            ts_obs = env._scene.get_observation()
            t = t + 1
        else:
            now_gpos = np.array(np.append(ts_obs.gripper_pose, ts_obs.gripper_open)) # 7 + 1 = 8
            
            target_gpos = now_gpos.copy()
            joyconrobotics.set_position = target_gpos[0:3]
        # 3.6954350499913655 Hz
        print(f'read rate = {1/(time.time() - frame_start_time)} Hz')
        if control_button != 0: # 按下录制下一个数据集，更新场景和机械臂，复位手柄
            if control_button == 1:
                recoder.do_run = False
                print(f'save the demo')
                recoder.join()  # 等待子线程结束
                demo = recoder.get_result()

            env.set_variation(0) 
            for i in range(len(path)):
                path[i].clear_visualization()
            _, ts_obs = env.reset() 
            
            joyconrobotics.set_position(right_zero_pos)
            joyconrobotics.orientation_sensor.reset_yaw()
            print(f'reset scene')
            
            if control_button == 1:
                processed_demo = Demo(demo)
                return processed_demo
            else:
                recoder = MyThread(target=recoding_process, args=(env._scene,))
                recoder.start()
        # print(f'the control frame rate = {(1/time.time() - frame_start_time)} Hz')
        time.sleep(0.01)
            

def init_joycon_rlbench(obs):
    # 获取到初始位置的末端位置
    gpos = np.array(np.append(obs.gripper_pose, obs.gripper_open)) # 7 + 1 = 8
    right_zero_pos = gpos[:3]
    right_zero_quat = gpos[3:7]
    right_zero_euler = R.from_quat(right_zero_quat).as_euler('xyz', degrees=True)
    # init_gpos = np.concatenate((right_zero_pos, right_zero_euler))
    # glimit = [[0.125, -0.4,  0.046, -3.1, -1.5, -1.5], 
    #          [0.380,  0.4,  0.23,  3.1,  1.5,  1.5]]
    name = "right"
    joyconrobotics = JoyconRobotics(
                        device=name, 
                        offset_position_m=right_zero_pos,
                        # offset_euler_rad=right_zero_euler,
                        pitch_down_double=False,
                        offset_euler_rad=[0, -math.pi, 0],
                        euler_reverse=[1, -1, 1],
                        direction_reverse=[-1, 1, 1],
                        pure_xz = False
                    )

    return joyconrobotics, right_zero_pos

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_demo(demo, example_path, ex_idx):
    data_dict = {
        '/action': [], 
        '/observations/images/wrist': [],
        # '/observations/images/wrist_depth': [], # not recommend
        '/observations/images/head': [],
        '/observations/qpos': [],
    }
    max_timesteps = len(demo)
    
    state2action_step = 1
    
    for i, obs in enumerate(demo): 
        if i >= state2action_step: 
            action = np.append(obs.joint_positions, obs.gripper_open)
            data_dict['/action'].append(action)
           
        data_dict['/observations/images/wrist'].append(obs.wrist_rgb) # 480， 640， 3
        data_dict['/observations/images/head'].append(obs.head_rgb)
        data_dict['/observations/qpos'].append(np.append(obs.joint_positions, obs.gripper_open))
        
    for idx in range(state2action_step):
        data_dict['/action'].append(np.append(obs.joint_positions, obs.gripper_open))
    
    action_len = np.shape(action)[0]
    
    dataset_path = os.path.join(example_path, f'episode_{ex_idx}') # save path
    check_and_make(example_path)
    
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root: 
        root.attrs['sim'] = True 
        action = root.create_dataset('action', (max_timesteps, action_len))
        obs = root.create_group('observations')
        image = obs.create_group('images')
        image.create_dataset('wrist', (max_timesteps, 480, 640, 3), dtype='uint8',chunks=(1, 480, 640, 3), ) # 640, 480 160, 120, 120, 160
        # image.create_dataset('wrist_depth', (max_timesteps, 480, 640, 3), dtype='uint8',chunks=(1, 480, 640, 3), ) 
        image.create_dataset('head', (max_timesteps, 480, 640, 3), dtype='uint8',chunks=(1, 480, 640, 3), ) 
        qpos = obs.create_dataset('qpos', (max_timesteps, action_len))
        
        for name, array in data_dict.items():
            root[name][...] = array
        print(f"demo save successfully: {dataset_path}")

def main(args):
    np.set_printoptions(linewidth=300)
    np.random.seed(10) 
    # command line parameters
    save_path = args['save_path']
    check_and_make(save_path)
    
    robot = args['robot']
    tasks = args['tasks']
    variations = args['variations'] - 1
    episodes_per_task = args['episodes_per_task']
    
    variation_path = os.path.join(save_path, tasks, VARIATIONS_FOLDER % 0)
    
    # tasks: sorting_program5, lamp_on, basketball_in_hoop
    task_env = make_sim_env(tasks, True, robot)
    env_max_reward = 1 # env.task.max_rewardz

    task_env.set_variation(variations) 
    descriptions, ts_obs = task_env.reset() 

    joyconrobotics, right_zero_pos = init_joycon_rlbench(ts_obs)
    
    for ex_idx in range(episodes_per_task):
        print('Process Task:', task_env.get_name(),
                  '// Variation:', variations, '// Demo:', ex_idx)
        demo = get_demo_joycon(joyconrobotics, task_env, right_zero_pos)
        save_demo(demo, variation_path, ex_idx + episodes_per_task * variations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default="Datasets" , help='dataste save path', required=True)
    parser.add_argument('--robot', type=str, default="sawyer" , help='robot name', required=True)
    parser.add_argument('--tasks', type=str, default="sorting_program5" , help='tasks name', required=True)
    parser.add_argument('--variations', type=int, default=1 , help='the variations of the task', required=True)
    parser.add_argument('--episodes_per_task', type=int, default=50 , help='the number of the episodes', required=True)
    
    main(vars(parser.parse_args()))