import os
import sys
import argparse
import numpy as np
import imageio
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up"
macros.IMAGE_CONVENTION = "opencv"

# fix cuda error
import os
os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.9/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.9"

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Now we can import the classes
from envs.franka_robosuite_tape_handover import FrankaRobosuiteTapeHandover
from envs.control.base_executor import CodeExecutionEnvBase, CodeExecEnvConfig
from api.franka_priviledged_api import FrankaControlTapeHandoverPrivilegedApi
from api.base_api import register_api

def parse_offset_list(offset_str):
    """
    Parse a comma-separated string of floats into a numpy array.
    
    Args:
        offset_str: String like "0.0,-0.7,0.0" or "0.0,0.7,0.0"
    
    Returns:
        numpy array of floats
    """
    try:
        values = [float(x.strip()) for x in offset_str.split(',')]
        if len(values) != 3:
            raise ValueError("Offset must contain exactly 3 values (x, y, z)")
        return np.array(values)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid offset format '{offset_str}': {e}. Expected format: 'x,y,z' (e.g., '0.0,-0.7,0.0')")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test handover step with configurable tape offsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_handover_step.py --yellow_offset 0.0,-0.7,0.0 --duct_offset 0.0,0.7,0.0
  python test_handover_step.py --yellow_offset 0.1,-0.6,0.0 --duct_offset -0.1,0.6,0.0
        """
    )
    parser.add_argument(
        '--yellow_offset',
        type=parse_offset_list,
        default='0.0,-0.7,0.0',
        help='Yellow tape offset as comma-separated x,y,z values (default: 0.0,-0.7,0.0)'
    )
    parser.add_argument(
        '--duct_offset',
        type=parse_offset_list,
        default='0.0,0.7,0.0',
        help='Duct tape offset as comma-separated x,y,z values (default: 0.0,0.7,0.0)'
    )
    
    args = parser.parse_args()
    
    # Extract offsets as numpy arrays
    yellow_offset = args.yellow_offset
    duct_offset = args.duct_offset
    
    print(f"Yellow tape offset: {yellow_offset}")
    print(f"Duct tape offset: {duct_offset}")
    
    # Register the API so CodeExecutionEnvBase can find it
    # The name here is used by CodeExecutionEnvBase to look up the API
    register_api("franka-handover-privileged", lambda env: FrankaControlTapeHandoverPrivilegedApi(env))

    # 1. Instantiate the low-level environment
    print("Initializing low-level FrankaRobosuiteTapeHandover environment...")
    low_level_env = FrankaRobosuiteTapeHandover(
        viser_debug=False,
        privileged=True,
        enable_render=False,
    )

    # 2. Define the configuration for the high-level code execution environment
    # We specify the low-level env and the API we just registered
    cfg = CodeExecEnvConfig(
        low_level=low_level_env,
        apis=["franka-handover-privileged"],
        prompt="Pick up the yellow tape with Arm 1 and hand it over to Arm 0.",
    )

    # 3. Instantiate the high-level environment
    # This environment's step() method takes a string of Python code
    print("Initializing high-level CodeExecutionEnvBase...")
    exec_env = CodeExecutionEnvBase(cfg)

    # 4. Enable video recording
    print("Enabling video capture...")
    exec_env.enable_video_capture(True)

    # 5. Reset the environment
    print("Resetting environment...")
    obs, info = exec_env.reset()

    # --- Manually set object position (relative offset) ---
    sim = low_level_env.robosuite_env.sim
    
    # Offsets are now passed in via command line arguments
    # Yellow tape
    yellow_tape_joint = low_level_env.robosuite_env.yellow_tape.joints[0]
    yellow_qpos = sim.data.get_joint_qpos(yellow_tape_joint).copy()
    yellow_qpos[:3] += yellow_offset
    sim.data.set_joint_qpos(yellow_tape_joint, yellow_qpos)

    # Duct tape
    duct_tape_joint = low_level_env.robosuite_env.duct_tape.joints[0]
    duct_qpos = sim.data.get_joint_qpos(duct_tape_joint).copy()
    duct_qpos[:3] += duct_offset
    sim.data.set_joint_qpos(duct_tape_joint, duct_qpos)

    sim.forward()
    # ------------------------------------

    # Format offsets for use in action code string
    yellow_offset_str = f"np.array([{yellow_offset[0]}, {yellow_offset[1]}, {yellow_offset[2]}])"
    duct_offset_str = f"np.array([{duct_offset[0]}, {duct_offset[1]}, {duct_offset[2]}])"
    
    action_code = f"""import numpy as np
import viser.transforms as vtf

# --- Get poses ---
yellow_tape_pos, yellow_tape_quat = get_object_pose("yellow tape")
yellow_tape_pos += {yellow_offset_str}
duct_tape_pos, duct_tape_quat = get_object_pose("duct tape")
duct_tape_pos += {duct_offset_str}

arm1_pos, _ = get_arm1_gripper_pose()
arm0_pos, _ = get_arm0_gripper_pose()
handover_pos = (arm1_pos + arm0_pos) / 2
arm0_handover_pos = handover_pos.copy()
# Need a way to get the width of the yellow tape that isnt privileged
# this is half the width of the franka gripper:
arm0_handover_pos[2] += 0.1025

# --- Pickup orientation ---
gripper_down_quat = np.array([0, 1, 0, 0])
gripper_side_matrix = vtf.SO3(wxyz=[0.707, 0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_side_quat = gripper_side_matrix.wxyz
gripper_rotated_side_matrix = vtf.SO3(wxyz=[0.707, -0.707, 0, 0]) @ vtf.SO3(wxyz=[0, 1, 0, 0])
gripper_rotated_side_quat = gripper_rotated_side_matrix.wxyz

# # Arm0 pick up duct tape
# open_gripper_arm0()
# goto_pose_arm0((duct_tape_pos+np.array([-0.01, -0.05, 0.0])), gripper_down_quat, z_approach=0.15)
# close_gripper_arm0()
# lifted = duct_tape_pos.copy(); lifted[2] = 0.15
# goto_pose_arm0(lifted, gripper_down_quat)
# goto_home_joint_position_arm0()

# Arm1: pick up yellow tape
open_gripper_arm1()
# shift the pick -y by 2cm to grab the tape on one end of the radius
goto_pose_arm1((yellow_tape_pos+np.array([-0.01, 0.05, -0.02])), gripper_down_quat, z_approach=0.15)
close_gripper_arm1()
lifted = yellow_tape_pos.copy(); lifted[2] = 0.15
goto_pose_arm1(lifted, gripper_down_quat)
goto_home_joint_position_arm1()

# Arm1: move to handover (shifted toward arm0)
goto_pose_arm1(handover_pos, gripper_rotated_side_quat)

# Arm0 approach
# arm0_quat = np.array([0.707, 0.707, 0, 0])
arm0_quat = gripper_side_quat
open_gripper_arm0()
# goto_pose_arm0(arm0_handover_pos + np.array([0.1, 0, 0.12]), arm0_quat, z_approach=0.1)
# goto_pose_arm0(arm0_handover_pos, arm0_quat, z_approach=0.12)
goto_pose_arm0(arm0_handover_pos + np.array([0, 0.02, 0]), arm0_quat, z_approach=0.10)
goto_pose_arm0(arm0_handover_pos + np.array([0, 0.02, 0]), arm0_quat, z_approach=0.01)
close_gripper_arm0()

# Arm1: release and retract
open_gripper_arm1()
shifted_handover_pos = handover_pos + vtf.SO3(wxyz=gripper_rotated_side_quat).as_matrix() @ np.array([0, 0, -0.1])
shifted_arm0_pos = arm0_handover_pos + vtf.SO3(wxyz=arm0_quat).as_matrix() @ np.array([0, 0, -0.1])
goto_pose_arm1(shifted_handover_pos, gripper_rotated_side_quat)
goto_pose_arm0(shifted_arm0_pos, arm0_quat)
goto_home_joint_position_arm1()
goto_home_joint_position_arm0()

# Arm0: drop cube in bowl, shifted to the left because the tape is slightly off-center in the robot's grasp
goto_pose_arm0((duct_tape_pos+np.array([-0.02, 0, 0.05])), gripper_down_quat, z_approach=0.15)
open_gripper_arm0()
goto_pose_arm0((duct_tape_pos+np.array([0, 0, 0.2])), gripper_down_quat)
goto_home_joint_position_arm0()
    """

    # 7. Run the step with the hardcoded action
    print("\nExecuting hardcoded action via exec_env.step()...")
    # This call triggers exec(action_code, ...) inside the executor
    obs, reward, terminated, truncated, info = exec_env.step(action_code)

    # 8. Save the recorded video
    video_frames = exec_env.get_video_frames()
    if video_frames:
        video_path = f"outputs/handover_video_{str(yellow_offset)}__{str(duct_offset)}.mp4"
        print(f"Saving video with {len(video_frames)} frames to {video_path}...")
        imageio.mimsave(video_path, video_frames, fps=20)
        print(f"Video saved to {video_path}")
    else:
        print("No video frames were captured.")

    # 9. Print execution results and logs
    print("\n" + "="*40)
    print("STEP EXECUTION RESULTS")
    print("="*40)
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Task Completed: {info.get('task_completed')}")
    
    print("\n--- STDOUT FROM CODE EXECUTION ---")
    print(info['stdout'])
    
    if info['stderr']:
        print("\n--- STDERR FROM CODE EXECUTION ---")
        print(info['stderr'])
    
    print("\nDone.")

if __name__ == "__main__":
    main()
