from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Import the above two first. Otherwise the cv2 window will not pop out

import cv2, numpy, torch, time
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from solution.checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from solution.checkpoint6 import CUBE_SIZE

cube_prompt = 'blue cube'
robot_ip = '192.168.1.155'

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by its 3D point cloud.
    """

    def __init__(self, camera_intrinsic):
        """
        Initialize the CubePoseDetector with camera parameters.

        Parameters
        ----------
        camera_intrinsic : numpy.ndarray
            The 3x3 intrinsic camera matrix.
        """
        self.camera_intrinsic = camera_intrinsic
        self.cube_size = CUBE_SIZE
        self.seg_threshold = 0.5

        # Initialize SAM3 model
        model = build_sam3_image_model(checkpoint_path='/home/rob/sam3/checkpoints/sam3.pt')
        self.seg_processor = Sam3Processor(model)

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : list or tuple
            A collection containing [image, point_cloud], where image is the 
            RGB/BGRA array and point_cloud is the registered 3D point cloud.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
            are 4x4 transformation matrices with translations in meters. 
            If no matching object is segmented, returns None.
        """
        image, point_cloud = observation

        # Get camera pose
        t_cam_robot = get_transform_camera_robot(image, self.camera_intrinsic)

        # Segment out the right cube
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        inference_state = self.seg_processor.set_image(image_tensor)
        output = self.seg_processor.set_text_prompt(state=inference_state, prompt=cube_prompt)

        if output['scores'].numel() == 0:
            print(f'Cannot detect {cube_prompt}')
            return None
        
        index = torch.argmax(output['scores'])
        mask = output['masks'][index].squeeze(0)

        # Filter out the cube's point cloud
        cube_points = point_cloud[mask.cpu()]
        valid_mask = ~numpy.isnan(cube_points[:, 0])
        cube_points = cube_points[valid_mask][:, :3] / 1000

        # Convert to the robot frame
        ones_column = numpy.ones((cube_points.shape[0], 1))
        cube_points_homogeneous = numpy.hstack((cube_points, ones_column))
        cube_points_robot = numpy.linalg.inv(t_cam_robot) @ cube_points_homogeneous.T
        cube_points_robot = cube_points_robot[:3, :].T

        cube_points_o3d = o3d.geometry.PointCloud()
        cube_points_o3d.points = o3d.utility.Vector3dVector(cube_points_robot)

        # Find the 3 normal vectors of the cube
        obb = cube_points_o3d.get_oriented_bounding_box()
        unit_vectors = obb.R

        # Determine the x unit vectors of the pose of the cube
        unit_x_robot = numpy.array([1, 0, 0])
        x_axis_angles = unit_x_robot @ unit_vectors   # The angle between the base x and every unit vectors
        index = numpy.argmax(numpy.abs(x_axis_angles))

        if x_axis_angles[index] < 0:
            x_unit = -unit_vectors[:, index]
        else:
            x_unit = unit_vectors[:, index]

        # Determine yaw angle and setup rotation matrix
        yaw = numpy.arctan2(x_unit[1], x_unit[0])
        euler_angles = [numpy.pi, 0, yaw]
        rotation_matrix = Rotation.from_euler('xyz', euler_angles).as_matrix()

        # Setup transformation matrix
        t_robot_cube = numpy.eye(4)
        t_robot_cube[:3, :3] = rotation_matrix
        t_robot_cube[:3, 3] = obb.center
        t_robot_cube[2, 3] = CUBE_SIZE

        # Determine the pose of the cube in camera frame
        t_cam_cube = t_cam_robot @ t_robot_cube

        return t_robot_cube, t_cam_cube

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

        # Get Transforms for cube
        results = cube_pose_detector.get_transforms([cv_image, point_cloud], cube_prompt)
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
