here are the prompts that somewhat worked: 


```
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
```
