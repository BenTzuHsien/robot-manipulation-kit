from solution.checkpoint3 import CubePoseDetector
# Import the above first. Otherwise the cv2 window will not pop out

import cv2, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from solution.checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

STACK_HEIGHT = 0.029

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

        # Get Transforms for Red Cube
        red_ok, green_ok = False, False
        red_results = cube_pose_detector.get_transforms(cv_image, 'red cube')
        if red_results is None:
            return
        red_t_robot_cube, red_t_cam_cube = red_results

        # Visualization
        red_cv_image = cv_image.copy()
        draw_pose_axes(red_cv_image, camera_intrinsic, red_t_cam_cube)
        cv2.namedWindow('Verifying Red Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Red Cube Pose', 1280, 720)
        cv2.imshow('Verifying Red Cube Pose', red_cv_image)
        key = cv2.waitKey(0)
    
        if key == ord('k'):
            cv2.destroyAllWindows()
            red_ok = True

        # Get Transforms for Red Cube
        green_results = cube_pose_detector.get_transforms(cv_image, 'green cube')
        if green_results is None:
            red_results
        green_t_robot_cube, green_t_cam_cube = green_results
        
        # Visualization
        green_cv_image = cv_image.copy()
        draw_pose_axes(green_cv_image, camera_intrinsic, green_t_cam_cube)
        cv2.namedWindow('Verifying Green Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Green Cube Pose', 1280, 720)
        cv2.imshow('Verifying Green Cube Pose', green_cv_image)
        key = cv2.waitKey(0)
    
        if key == ord('k'):
            cv2.destroyAllWindows()
            green_ok = True

        # Execute Stacking Task
        if red_ok and green_ok:

            # Grasp Red Cube
            grasp_cube(arm, red_t_robot_cube)

            # Stack on Top of Green Cube
            green_t_robot_cube[2, 3] += STACK_HEIGHT
            place_cube(arm, green_t_robot_cube)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
