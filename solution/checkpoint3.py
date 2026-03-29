from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Import the above two first. Otherwise the cv2 window will not pop out

import cv2, numpy, torch, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from solution.checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

cube_prompt = 'blue cube'
robot_ip = '192.168.1.155'

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by the AprilTags.
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
        self.tag_size = CUBE_TAG_SIZE
        self.seg_threshold = 0.5

        # Initialize AprilTag Detector
        self.april_tag_detector = Detector(families=CUBE_TAG_FAMILY)

        # Initialize SAM3 model
        model = build_sam3_image_model(checkpoint_path='/home/rob/sam3/checkpoints/sam3.pt')
        self.seg_processor = Sam3Processor(model)

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : numpy.ndarray
            The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
            are 4x4 transformation matrices with translations in meters. 
            If no matching object or tag is found, returns None.
        """
        # Segment out the right cube
        observation_rgb = cv2.cvtColor(observation, cv2.COLOR_BGRA2RGB)
        observation_tensor = torch.from_numpy(observation_rgb).permute(2, 0, 1)
        inference_state = self.seg_processor.set_image(observation_tensor)
        output = self.seg_processor.set_text_prompt(state=inference_state, prompt=cube_prompt)

        if output['scores'].numel() == 0:
            print(f'Cannot detect {cube_prompt}')
            return None
        
        cube_bboxes = []
        for index, score in enumerate(output['scores']):
            if score > self.seg_threshold:
                cube_bboxes.append(output['boxes'][index])

        # Detect AprilTag Points
        if len(observation.shape) > 2:
            observation_gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
        tags = self.april_tag_detector.detect(observation_gray, 
                                              estimate_tag_pose=True, 
                                              camera_params=[self.camera_intrinsic[0, 0],
                                                             self.camera_intrinsic[1, 1],
                                                             self.camera_intrinsic[0, 2],
                                                             self.camera_intrinsic[1, 2]],
                                                             tag_size=self.tag_size)

        cube_tags = [t for t in tags if t.tag_id == CUBE_TAG_ID]

        # Check which tag is inside cube' bbox
        for bbox in cube_bboxes:
            for tag in cube_tags:
                tag_x, tag_y = tag.center
                if (tag_x > bbox[0]) and (tag_x < bbox[2]) and (tag_y > bbox[1]) and (tag_y < bbox[3]):

                    frame_rotation = numpy.array([[0, 1, 0],
                                                [-1, 0, 0],
                                                [0, 0, 1]])
                    t_cam_cube = numpy.eye(4)
                    t_cam_cube[:3, :3] = tag.pose_R @ frame_rotation
                    t_cam_cube[:3, 3] = tag.pose_t.flatten()

                    t_cam_robot = get_transform_camera_robot(observation, self.camera_intrinsic)
                    t_robot_cube = numpy.linalg.inv(t_cam_robot) @ t_cam_cube

                    return t_robot_cube, t_cam_cube
        
        print(f'No {cube_prompt} has AprilTag.')
        return None

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

        # Get Transforms for cube
        results = cube_pose_detector.get_transforms(cv_image, cube_prompt)
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
