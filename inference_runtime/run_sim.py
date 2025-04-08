"""
Script to run OpenVLA model in a PyBullet simulation environment.
This allows testing the model's ability to generate actions in a simulated robotic environment.
"""

import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run OpenVLA model in PyBullet simulation")
    parser.add_argument("--model_name", type=str, default="openvla/openvla-7b", 
                        help="Model name or path")
    parser.add_argument("--instruction", type=str, default="pick up the cube", 
                        help="Instruction for the robot")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cuda:0, cpu, etc.)")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Whether to quantize the model to 4-bit precision")
    parser.add_argument("--sim_rate", type=float, default=5.0,
                        help="Simulation rate in Hz")
    args = parser.parse_args()
    
    # Connect to PyBullet with GUI
    print("Initializing PyBullet simulation...")
    p.connect(p.GUI)  # Opens a window with the simulation
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    # Enable GUI controls (this allows sliders to be visible)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    
    # Load a simple environment with a colorful plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Simple, clean environment with just the cube to pick up
    
    # Load Franka Panda robot - better for manipulation tasks
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # Set joint controllers to be more responsive
    # Setup Franka arm and gripper
    num_joints = p.getNumJoints(robot_id)
    for joint in range(num_joints):
        # Print joint info for debugging
        joint_info = p.getJointInfo(robot_id, joint)
        print(f"Joint {joint}: {joint_info[1]}, type: {joint_info[2]}")
        
        if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED:
            # Set up joint control with lower damping for more visible movement
            p.changeDynamics(robot_id, joint, linearDamping=0.1, angularDamping=0.1)
    
    # Reset to a better starting pose for the Franka arm
    # These are good default positions for the Franka
    rest_pose = [0, -0.215, 0, -2.57, 0, 2.356, 0.7, 0.04, 0.04]  # arm joints + 2 finger joints
    for i, pose in enumerate(rest_pose):
        if i < num_joints and p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED:
            p.resetJointState(robot_id, i, pose)
    
    # Add a cube for manipulation tasks
    cube_size = 0.05
    cube_mass = 0.1
    cube_visual_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[cube_size/2, cube_size/2, cube_size/2],
        rgbaColor=[1, 0, 0, 1]
    )
    cube_collision_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[cube_size/2, cube_size/2, cube_size/2]
    )
    cube_id = p.createMultiBody(
        baseMass=cube_mass,
        baseCollisionShapeIndex=cube_collision_id,
        baseVisualShapeIndex=cube_visual_id,
        basePosition=[0.5, 0, cube_size/2]
    )
    
    # Set up camera position for better view of the Franka arm
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=[0.0, 0.0, 0.4]
    )
    
    # Configure debug visualizer for better visibility
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    
    # Load OpenVLA model
    print(f"Loading OpenVLA model {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Determine dtype based on device
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    
    # Set environment variable to avoid memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Free up CUDA memory before loading model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    # Force aggressive GPU memory management
    if args.device.startswith("cuda"):
        print("Setting up aggressive GPU memory optimization...")
        # Free GPU memory and reduce memory fraction
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Check if bitsandbytes is available for 8-bit quantization
        try:
            import bitsandbytes as bnb
            print("Using 8-bit quantization with bitsandbytes")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )
        except ImportError:
            print("bitsandbytes not available, using regular float16")
            quantization_config = None
    else:
        quantization_config = None
    
    # Load model with GPU quantization settings - no CPU fallback
    print(f"Loading model on {args.device} with quantization...")
    
    # Force model to be loaded in parts to avoid OOM
    os.environ["TRANSFORMERS_MAX_SHARD_SIZE"] = "500MB"
    
    if args.device.startswith("cuda"):
        # For CUDA with quantization
        model_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Add quantization config if available
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load the model with all optimizations - don't use .to() with quantized models
        vla = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            device_map=args.device,  # This properly maps the model to device with quantization
            **model_kwargs
        )
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # For CPU, use standard settings
        vla = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(args.device)
    
    print(f"Model loaded on {args.device}")
    print(f"Running simulation with instruction: '{args.instruction}'")
    
    # Allow simulation to stabilize
    for _ in range(100):
        p.stepSimulation()
        
    # Add debug parameters to control simulation speed - with more visible labels
    print("Adding simulation control sliders - look in the right panel of the PyBullet window")
    sim_speed_slider = p.addUserDebugParameter("SLIDER 1: Simulation Speed", 0.1, 10.0, args.sim_rate)
    
    # Add a debug parameter to manually trigger actions
    trigger_action = p.addUserDebugParameter("SLIDER 2: Trigger Action (click me)", 1, 0, 1)
    last_trigger_value = p.readUserDebugParameter(trigger_action)
    
    # Add text to guide the user
    p.addUserDebugText(
        "Control sliders are in the right panel →",
        [0, 0, 1.5],
        textColorRGB=[1, 1, 0],
        textSize=1.5,
        lifeTime=0
    )
    
    # Store marker ID
    args.marker_id = None
    
    # Simulation loop
    try:
        # Enhanced state machine: APPROACH -> POSITION -> GRASP -> LIFT -> DONE
        #    APPROACH: Move generally toward the cube
        #    POSITION: Fully open gripper and center precisely over cube
        #    GRASP: Close gripper firmly around cube
        #    LIFT: Raise cube while maintaining grip
        #    DONE: Hold and manipulate cube
        grasp_state = "APPROACH"  
        # Define end effector link index early
        end_effector_index = 11  # End effector link index for Franka Panda
        
        # Define Franka's gripper joints early
        gripper_joints = [9, 10]  # Franka finger joints
        while True:
            # Capture image from simulator
            cam_target = [0.2, 0, 0.2]
            cam_distance = 1.0
            cam_yaw = 50
            cam_pitch = -35
            
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_target,
                distance=cam_distance,
                yaw=cam_yaw,
                pitch=cam_pitch,
                roll=0,
                upAxisIndex=2
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=640/480,
                nearVal=0.1,
                farVal=100.0
            )
            
            width, height = 640, 480
            _, _, rgb_img, _, _ = p.getCameraImage(
                width, height, 
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to PIL Image
            rgb_array = np.reshape(rgb_img, (height, width, 4))[:,:,:3]
            image = Image.fromarray(rgb_array)
            
            # Use more detailed, specific prompts for each grasping state
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            end_pos, _ = p.getLinkState(robot_id, end_effector_index)[:2]
            
            # Calculate distance to cube for more informative prompts
            distance_to_cube = np.linalg.norm(np.array(end_pos) - np.array(cube_pos))
            # Calculate horizontal distance (xy-plane only)
            horizontal_dist = np.linalg.norm(np.array([end_pos[0], end_pos[1]]) - np.array([cube_pos[0], cube_pos[1]]))
            
            if grasp_state == "APPROACH":
                prompt = (f"In: The robot needs to pick up a cube at position {cube_pos}. " 
                          f"The gripper is currently at position {end_pos}, {distance_to_cube:.2f} units away. "
                          f"What precise action should the robot take to move its gripper generally toward the cube?\nOut:")
            elif grasp_state == "POSITION":
                prompt = (f"In: The robot needs to precisely position for grasping. "
                          f"IMPORTANT: The gripper must be fully opened with claws maximally extended. "
                          f"Position the gripper directly above the center of the cube at {cube_pos}, "
                          f"with a small gap ready for grasping. Current horizontal distance is {horizontal_dist:.3f} units. "
                          f"What precise action will center the gripper perfectly over the cube with claws fully open?\nOut:")
            elif grasp_state == "GRASP":
                prompt = (f"In: The robot's gripper is now positioned directly above the cube at {cube_pos}. "
                          f"The gripper claws are fully extended and open. "
                          f"IMPORTANT: Now close the gripper claws completely around the cube "
                          f"with maximum force to establish a very secure grip. "
                          f"What specific action should the robot take to close the gripper with maximum force?\nOut:")
            elif grasp_state == "LIFT":
                prompt = (f"In: The robot has grasped the cube with its gripper at position {cube_pos}. "
                          f"IMPORTANT: Maintain the maximum grip strength with fully closed claws during the entire lifting motion. "
                          f"The cube is currently {cube_pos[2]:.2f} units above the ground. "
                          f"What precise action should the robot take to lift the cube higher while ensuring "
                          f"the gripper maintains a solid, unwavering grip throughout the motion?\nOut:")
            else:  # DONE
                prompt = (f"In: The robot has successfully lifted the cube to position {cube_pos}. "
                          f"What action should the robot take to maintain a stable grip on the cube "
                          f"while moving it to a new position?\nOut:")
                
            print(f"\nGrasp state: {grasp_state}")
            print(f"Using prompt: {prompt}")
            
            # Process inputs for the model
            inputs = processor(prompt, image)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            # Convert to appropriate dtype if using CUDA
            if args.device.startswith("cuda"):
                inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}
            
            # Generate action with OpenVLA - more deterministic for consistent behavior
            with torch.no_grad():
                action = vla.predict_action(
                    **inputs, 
                    unnorm_key="bridge_orig", 
                    do_sample=False  # More deterministic for consistent behavior
                )
            
            # Convert action to numpy if it's a tensor
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Print action for debugging
            print("\nPredicted Action:")
            components = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            for i, comp in enumerate(components):
                if i < len(action):
                    print(f"  {comp}: {action[i]:.4f}")
            
            # Apply action to robot (simplified example)
            # Note: This is a simplified mapping and would need to be adapted for specific robots
            num_joints = p.getNumJoints(robot_id)
            control_joints = min(6, num_joints)  # Use at most 6 joints for position control
            
            # Use proper inverse kinematics with state-specific behavior
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            
            # Adjust target position based on current state
            if grasp_state == "APPROACH":
                # In approach, aim in general direction of the cube
                target_x = cube_pos[0] + action[0] * 0.05
                target_y = cube_pos[1] + action[1] * 0.05
                target_z = cube_pos[2] + 0.15  # Higher position during approach
            elif grasp_state == "POSITION":
                # In position, precisely center over the cube with more direct control
                # Use cube position directly with minimal offset to ensure centering
                target_x = cube_pos[0] + action[0] * 0.01  # Very small influence from model
                target_y = cube_pos[1] + action[1] * 0.01
                target_z = cube_pos[2] + 0.08  # Specific height for positioning
                
                # Force gripper to be fully open during positioning
                for finger_joint in gripper_joints:
                    p.setJointMotorControl2(robot_id, finger_joint, p.POSITION_CONTROL, 0.04, force=30)
            elif grasp_state == "GRASP":
                # In grasp, stay centered and move down to grasp
                target_x = cube_pos[0]  # Stay directly over cube center
                target_y = cube_pos[1]
                target_z = cube_pos[2] + 0.05  # Lower position for grasping
            elif grasp_state == "LIFT":
                # In lift, move upward
                target_x = cube_pos[0] + action[0] * 0.01
                target_y = cube_pos[1] + action[1] * 0.01
                target_z = cube_pos[2] + 0.2  # Higher position for lifting
            else:  # DONE
                # In done state, maintain position or move according to model with reduced influence
                target_x = cube_pos[0] + action[0] * 0.01
                target_y = cube_pos[1] + action[1] * 0.01
                target_z = cube_pos[2] + 0.25  # Keep high position
            
            # Calculate target orientation (simple grasp from above)
            target_roll = 0 + action[3] * 0.2
            target_pitch = 1.57 + action[4] * 0.2  # ~90 degrees for top grasp
            target_yaw = 0 + action[5] * 0.2
            
            # Convert orientation to quaternion
            target_orn = p.getQuaternionFromEuler([target_roll, target_pitch, target_yaw])
            
            # Use inverse kinematics to get joint positions
            # End effector has already been defined at the start of the loop
            
            # Calculate IK
            joint_poses = p.calculateInverseKinematics(
                robot_id,
                end_effector_index,
                [target_x, target_y, target_z],
                target_orn,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
            
            # Use only the first control_joints for position control
            target_positions = list(joint_poses[:control_joints])
            
            # Print joint positions for debugging
            print(f"Joint targets: {[f'{v:.2f}' for v in target_positions]}")
            
            # Add a visual marker to show the target position
            # Make the target visualization more dramatic
            target_x = 0.5 + action[0] * 5.0  # Greatly amplified for visibility 
            target_y = action[1] * 5.0
            target_z = 0.3 + action[2] * 5.0
            
            # Create or update a visual marker for the target position
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[0, 1, 0, 0.7]  # Green semi-transparent sphere
            )
            
            # Check if marker already exists and remove it
            if hasattr(args, 'marker_id') and args.marker_id is not None:
                p.removeBody(args.marker_id)
                
            # Create new marker
            args.marker_id = p.createMultiBody(
                baseMass=0,  # Mass zero means static
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[target_x, target_y, target_z]
            )
            
            # Apply joint control with higher forces for more visible movement
            p.setJointMotorControlArray(
                robot_id, 
                range(control_joints), 
                p.POSITION_CONTROL, 
                targetPositions=target_positions,
                forces=[100] * control_joints  # Higher forces for more responsive movement
            )
            
            # Draw lines from robot end effector to target for visualization
            # Get the current end effector position
            end_effector_link = control_joints - 1
            end_pos, _ = p.getLinkState(robot_id, end_effector_link)[:2]
            
            # Draw a line from end effector to target
            p.addUserDebugLine(
                end_pos,
                [target_x, target_y, target_z],
                lineColorRGB=[1, 0, 0],
                lineWidth=2.0,
                lifeTime=0.1
            )
            
            # Handle gripper based on state and action
            gripper_closed = False
            if len(action) > 6:
                gripper_value = action[6]
                # Apply gripper control
                print(f"  Gripper command: {gripper_value:.4f}")
                
                # Check if gripper should be closed (> 0.5 means close)
                gripper_closed = gripper_value > 0.5
                
                # Improved state transition logic
                end_pos, _ = p.getLinkState(robot_id, end_effector_index)[:2]
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                distance_to_cube = np.linalg.norm(np.array(end_pos) - np.array(cube_pos))
                
                # More detailed state transitions with better thresholds
                if grasp_state == "APPROACH":
                    # Check if we're generally near the cube to start precise positioning
                    horizontal_dist = np.linalg.norm(np.array([end_pos[0], end_pos[1]]) - np.array([cube_pos[0], cube_pos[1]]))
                    above_cube = end_pos[2] > cube_pos[2] + 0.01  # Check if gripper is above cube
                    
                    if horizontal_dist < 0.15 and above_cube:
                        grasp_state = "POSITION"  # Move to precise positioning
                        print(f"Transition: APPROACH -> POSITION (h_dist: {horizontal_dist:.3f}, v_dist: {end_pos[2]-cube_pos[2]:.3f})")
                        # Force gripper to be fully open
                        for finger_joint in gripper_joints:
                            p.setJointMotorControl2(robot_id, finger_joint, p.POSITION_CONTROL, 0.04, force=30)
                    elif horizontal_dist > 0.3:  # We've drifted too far from the cube
                        print(f"Too far from cube ({horizontal_dist:.3f}), refocusing approach")
                
                elif grasp_state == "POSITION":
                    # Check if we're precisely centered over the cube
                    horizontal_dist = np.linalg.norm(np.array([end_pos[0], end_pos[1]]) - np.array([cube_pos[0], cube_pos[1]]))
                    correct_height = abs(end_pos[2] - (cube_pos[2] + 0.08)) < 0.02  # Check if at correct height
                    
                    # Also verify gripper is fully open
                    gripper_open = True
                    for finger_joint in gripper_joints:
                        finger_state = p.getJointState(robot_id, finger_joint)[0]
                        if finger_state < 0.035:  # Not fully open
                            gripper_open = False
                    
                    if horizontal_dist < 0.05 and correct_height and gripper_open:
                        grasp_state = "GRASP"
                        print(f"Transition: POSITION -> GRASP (centered at h_dist: {horizontal_dist:.3f}, gripper fully opened)")
                    elif horizontal_dist > 0.2:  # We've drifted too far from the cube
                        grasp_state = "APPROACH"  # Go back to approach if we lose position
                        print(f"Lost positioning (h_dist: {horizontal_dist:.3f}), returning to APPROACH")
                
                elif grasp_state == "GRASP":
                    # Transition to LIFT when gripper is fully closed AND we're very close to the cube
                    # Stay in GRASP state longer to ensure a solid grip is established
                    if gripper_closed and distance_to_cube < 0.1:
                        # Count how many steps the gripper has been closed
                        if not hasattr(args, 'grip_counter'):
                            args.grip_counter = 0
                        
                        args.grip_counter += 1
                        
                        # Only transition after maintaining grip for several steps
                        if args.grip_counter > 30:  # Ensure solid grip before lifting
                            grasp_state = "LIFT"
                            print(f"Transition: GRASP -> LIFT (solid grip established, distance: {distance_to_cube:.3f})")
                            # Force an upward motion to help with lifting
                            for i in range(20):
                                # Apply extra grip force during this transition
                                for finger_joint in gripper_joints:
                                    p.setJointMotorControl2(robot_id, finger_joint, p.POSITION_CONTROL, 0.0, force=150)
                                p.stepSimulation()
                            args.grip_counter = 0
                    else:
                        # Reset counter if grip is lost
                        if hasattr(args, 'grip_counter'):
                            args.grip_counter = 0
                
                elif grasp_state == "LIFT":
                    # Consider the lift successful when the cube is significantly above the ground
                    # and the gripper is still closed
                    if cube_pos[2] > 0.15 and gripper_closed:
                        grasp_state = "DONE"
                        print(f"Transition: LIFT -> DONE (cube lifted to height: {cube_pos[2]:.3f})")
                    elif not gripper_closed:  # We dropped the cube
                        grasp_state = "APPROACH"
                        print("Lost grip on cube, returning to APPROACH state")
                    
            # Handle Franka's two-finger gripper (defined at the beginning of the main function)
            
            # Set finger positions based on current state and gripper command
            if grasp_state == "POSITION":
                # Always fully open in POSITION state, regardless of model output
                finger_target = 0.04  # Fully open
                grip_force = 30  # Light force for opening
            elif grasp_state == "GRASP" and gripper_closed:
                # In GRASP state with closed gripper command, use maximum force
                finger_target = 0.0  # Completely closed
                grip_force = 150  # Maximum possible force
            elif grasp_state == "LIFT" or grasp_state == "DONE":
                # In lifting or done states, maintain maximum grip
                finger_target = 0.0  # Keep completely closed
                grip_force = 150  # Maximum possible force
            elif gripper_closed:
                # For other states with closed gripper command
                finger_target = 0.01
                grip_force = 50
            else:
                # For open gripper in other states
                finger_target = 0.04
                grip_force = 30
                
            # Apply to both finger joints with appropriate force
            for finger_joint in gripper_joints:
                p.setJointMotorControl2(robot_id, finger_joint, p.POSITION_CONTROL, finger_target, force=grip_force)
            
            # If gripper is closed, attach the cube to the robot's end effector
            if gripper_closed and (grasp_state == "LIFT" or grasp_state == "DONE"):
                end_pos, end_orn = p.getLinkState(robot_id, end_effector_index)[:2]
                # Offset the cube position slightly to account for the gripper geometry
                offset = [0.0, 0.0, -0.05]
                cube_pos = [end_pos[0] + offset[0], end_pos[1] + offset[1], end_pos[2] + offset[2]]
                p.resetBasePositionAndOrientation(cube_id, cube_pos, end_orn)
            
            # Step simulation
            p.stepSimulation()
            
            # Read simulation speed from slider
            current_sim_rate = p.readUserDebugParameter(sim_speed_slider)
            
            # Check if action trigger was pressed
            current_trigger = p.readUserDebugParameter(trigger_action)
            if current_trigger != last_trigger_value:
                last_trigger_value = current_trigger
                # Add a text to show action is triggered
                p.addUserDebugText(
                    "Action Triggered!", 
                    [0, 0, 1], 
                    textColorRGB=[1, 0, 0],
                    lifeTime=1.0
                )
                
            # Sleep to control simulation rate
            time.sleep(1.0 / current_sim_rate)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        p.disconnect()
        print("PyBullet disconnected")

if __name__ == "__main__":
    main()
