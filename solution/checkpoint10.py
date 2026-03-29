from solution.checkpoint8 import CubePoseDetector
# Import the above first. Otherwise the cv2 window will not pop out

import cv2, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from solution.checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from solution.checkpoint4 import STACK_HEIGHT

stacking_order = ['red cube', 'green cube', 'blue cube']   # From top to bottom
robot_ip = '192.168.1.155'

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Cube Pose Detector
    cube_pose_detector = CubePoseDetector(camera_intrinsic)

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud

        # Initialize a List to Store Cube Pose
        cube_poses = []
        
        # Get Transforms for Cubes
        for cube_prompt in stacking_order:
            results = cube_pose_detector.get_transforms([cv_image, point_cloud], cube_prompt)

            # Check Results
            if results is None:
                continue
            t_robot_cube, t_cam_cube = results

            cv_image_current = cv_image.copy()
            draw_pose_axes(cv_image_current, camera_intrinsic, t_cam_cube)
            cv2.namedWindow(f'Verifying {cube_prompt} Pose', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Verifying {cube_prompt} Pose', 1280, 720)
            cv2.imshow(f'Verifying {cube_prompt} Pose', cv_image_current)
            key = cv2.waitKey(0)

            if key == ord('k'):
                cv2.destroyAllWindows()
                cube_poses.append(t_robot_cube.copy())
        
        # Execute Stacking Task
        if len(cube_poses) == 3:

            # Grasp Second Cube
            grasp_cube(arm, cube_poses[1])

            # Stack on Top of Third Cube
            cube_poses[2][2, 3] += STACK_HEIGHT
            place_cube(arm, cube_poses[2])

            # Grasp First Cube
            grasp_cube(arm, cube_poses[0])

            # Stack on Top of Second Cube
            cube_poses[2][2, 3] += STACK_HEIGHT
            place_cube(arm, cube_poses[2])
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
