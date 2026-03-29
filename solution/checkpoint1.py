import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

GRIPPER_LENGTH = 0.067 * 1000
CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0207

robot_ip = '192.168.1.155'

def grasp_cube(arm, cube_pose, vaccum_gripper=False, grasp_lower_step=0.005):
    """
    Execute a pick sequence to grasp a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the cube's pose in the robot base frame.
        All translational units in this matrix are in meters.
    vaccum_gripper : bool, optional
        If True, uses the vacuum gripper logic instead of the standard Lite6 
        gripper. Defaults to False.
    grasp_lower_step : float, optional
        The distance in meters to incrementally lower the vacuum gripper if 
        suction fails on the first attempt. Defaults to 0.005.
    """
    rot = Rotation.from_matrix(cube_pose[:3, :3])
    roll, pitch, yaw = rot.as_euler('xyz')

    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000
    z_pregrasp = z + 50
    z_hold = z + 100

    # Pregrasp Location
    arm.set_position(x, y, z_pregrasp, roll, pitch, yaw, is_radian=True, wait=True)
    if not vaccum_gripper:
        arm.open_lite6_gripper()
    time.sleep(0.5)

    # Grasp Location
    arm.set_position(x, y, z, roll, pitch, yaw, is_radian=True, wait=True)
    time.sleep(0.5)

    # Grasp
    if not vaccum_gripper:
        arm.close_lite6_gripper()
        time.sleep(1)
        arm.stop_lite6_gripper()
    
    else:
        arm.set_vacuum_gripper(on=True, wait=True)
        time.sleep(1)

        # Check grasp
        code, state = arm.get_vacuum_gripper()
        while code != 0 or state != 1:
            z -= grasp_lower_step * 1000
            arm.set_position(x, y, z, roll, pitch, yaw, is_radian=True, wait=True)
            time.sleep(0.5)
            code, state = arm.get_vacuum_gripper()

    # Hold the cube high
    arm.set_position(x, y, z_hold, roll, pitch, yaw, is_radian=True, wait=True)
    time.sleep(0.5)

def place_cube(arm, cube_pose, vaccum_gripper=False):
    """
    Execute a place sequence to release a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the target placement pose in the robot base frame.
        All translational units in this matrix are in meters.
    vaccum_gripper : bool, optional
        If True, uses the vacuum gripper logic instead of the standard Lite6 
        gripper. Defaults to False.
    """
    rot = Rotation.from_matrix(cube_pose[:3, :3])
    roll, pitch, yaw = rot.as_euler('xyz')

    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000
    z_postplace = z + 50
    
    # Place Location
    arm.set_position(x, y, z, roll, pitch, yaw, is_radian=True, wait=True)
    time.sleep(0.5)

    # Place
    if not vaccum_gripper:
        arm.open_lite6_gripper()
        time.sleep(1)
        arm.stop_lite6_gripper()
    
    else:
        arm.set_vacuum_gripper(on=False, wait=True)
        time.sleep(1)

    # Postplace Location
    arm.set_position(x, y, z_postplace, roll, pitch, yaw, is_radian=True, wait=True)
    time.sleep(0.5)

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function uses visual fiducial detection to find the cube's pose in the camera's view, 
    then transforms that pose into the robot's global coordinate system. 

    Parameters
    ----------
    observation : numpy.ndarray
        The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix representing the camera's pose in the robot base frame (t_cam_robot).
        All translations are in meters.

    Returns
    -------
    tuple or None
        If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
        are 4x4 transformation matrices with translations in meters. 
        If no cube tag is detected, returns None.
    """
    # Initialize AprilTag Detector
    detector = Detector(families=CUBE_TAG_FAMILY)

    # Detect AprilTag Points
    if len(observation.shape) > 2:
        observation = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    tags = detector.detect(observation, estimate_tag_pose=True, camera_params=[camera_intrinsic[0, 0], 
                                                                               camera_intrinsic[1, 1], 
                                                                               camera_intrinsic[0, 2], 
                                                                               camera_intrinsic[1, 2]], tag_size=CUBE_TAG_SIZE)

    cube_tag = next((t for t in tags if t.tag_id == CUBE_TAG_ID), None)
    if cube_tag is None:
        print("Cube tag not found.")
        return None

    frame_rotation = numpy.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = cube_tag.pose_R @ frame_rotation
    t_cam_cube[:3, 3] = cube_tag.pose_t.flatten()

    t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube

    return t_robot_cube, t_cam_cube

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

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

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
        # Get Transform Between Camera and Cube
        results = get_transform_cube(cv_image, camera_intrinsic, t_cam_robot)
        if results is None:
            return
        t_robot_cube, t_cam_cube = results
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # Grasp the cube
            grasp_cube(arm, t_robot_cube)

            # Place the Cube
            place_cube(arm, t_robot_cube)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
