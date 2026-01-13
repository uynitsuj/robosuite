import os
import sys
import numpy as np

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Now we can import the classes
from envs.franka_robosuite_tape_handover import FrankaRobosuiteTapeHandover
from envs.control.base_executor import CodeExecutionEnvBase, CodeExecEnvConfig
from api.franka_priviledged_api import FrankaControlTapeHandoverPrivilegedApi
from api.base_api import register_api

def main():
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
        prompt="Pick up the yellow tape with Arm 0 and hand it over to Arm 1.",
    )

    # 3. Instantiate the high-level environment
    # This environment's step() method takes a string of Python code
    print("Initializing high-level CodeExecutionEnvBase...")
    exec_env = CodeExecutionEnvBase(cfg)

    # 4. Reset the environment
    print("Resetting environment...")
    obs, info = exec_env.reset()

    # 5. Define a hardcoded action (Python code as a string)
    # This code will be executed in a namespace where API functions are globally available
    action_code = """
print("--- Starting robot execution script ---")

# Step 1: Get the pose of the yellow tape (privileged info)
pos, quat = get_object_pose("yellow tape")
print(f"Detected yellow tape at: {pos}")

# Step 2: Prepare the gripper
print("Opening Arm 0 gripper...")
open_gripper_arm0()

# Step 3: Move to approach pose (5cm above the tape)
print("Moving to approach pose...")
goto_pose_arm0(pos, quat, z_approach=0.05)

# Step 4: Move to the tape
print("Moving to grasp pose...")
goto_pose_arm0(pos, quat)

# Step 5: Grasp the tape
print("Closing Arm 0 gripper...")
close_gripper_arm0()

# Step 6: Lift the tape
print("Lifting tape...")
goto_pose_arm0(pos + [0, 0, 0.15], quat)

print("--- Robot execution script finished ---")
RESULT = "Tape picked and lifted"
"""

    # 6. Run the step with the hardcoded action
    print("\nExecuting hardcoded action via exec_env.step()...")
    # This call triggers exec(action_code, ...) inside the executor
    obs, reward, terminated, truncated, info = exec_env.step(action_code)

    # 7. Print execution results and logs
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
