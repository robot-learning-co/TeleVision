import math
import numpy as np

np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
from constants_vuer import *
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
import pyzed.sl as sl
from dynamixel.active_cam import DynamixelAgent
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from motion_utils import fast_mat_inv


resolution = (720, 1280)
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

agent = DynamixelAgent(port="/dev/ttyACM0")
agent._robot.set_torque_mode(True)

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 opr HD1200 video mode, depending on camera type.
init_params.camera_fps = 60  # Set fps at 60

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(err) + ". Exit program.")
    exit()

# Capture 50 frames and stop
i = 0
image_left = sl.Mat()
image_right = sl.Mat()
runtime_parameters = sl.RuntimeParameters()

img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
img_height, img_width = resolution_cropped[:2]
shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming, tunnel=True)

# processor = VuerPreprocessor()

while True:
    start = time.time()
    
    # head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = processor.process(tv)

    # HEAD
    head_mat = grd_yup2grd_zup[:3, :3] @ tv.head_matrix[:3, :3] @ grd_yup2grd_zup[:3, :3].T
    if np.sum(head_mat) == 0:
        head_mat = np.eye(3)
    head_rot = rotations.quaternion_from_matrix(head_mat[0:3, 0:3])
    
    # HANDS
    left_handmarks = tv.left_landmarks
    right_handmarks = tv.right_landmarks
    left_state = tv.left_state
    right_state = tv.right_state

    # Track pinch start positions and movements
    if left_state[0] and not hasattr(tv, 'left_pinch_start'):
        # Record initial position when pinch starts
        tv.left_pinch_start = left_handmarks[0].copy()
        print("Left pinch started")
    elif not left_state[0] and hasattr(tv, 'left_pinch_start'):
        # Clear pinch start position when pinch ends
        delattr(tv, 'left_pinch_start')
        print("Left pinch ended")
    
    if right_state[0] and not hasattr(tv, 'right_pinch_start'):
        # Record initial position when pinch starts 
        tv.right_pinch_start = right_handmarks[0].copy()
        print("Right pinch started")
    elif not right_state[0] and hasattr(tv, 'right_pinch_start'):
        # Clear pinch start position when pinch ends
        delattr(tv, 'right_pinch_start')
        print("Right pinch ended")

    # Print relative movements during pinch
    if left_state[0] and hasattr(tv, 'left_pinch_start'):
        rel_movement = left_handmarks[0] - tv.left_pinch_start
        print(f"Left hand relative movement: {rel_movement[3,:3]}")
        
    if right_state[0] and hasattr(tv, 'right_pinch_start'):
        rel_movement = right_handmarks[0] - tv.right_pinch_start
        print(f"Right hand relative movement: {rel_movement[3,:3]}")
    
    # # if pinch
    # if left_state[0]:
    #     left_hand_mat = grd_yup2grd_zup @ left_handmarks[0] @ fast_mat_inv(grd_yup2grd_zup)
    #     # left_hand_mat = grd_yup2grd_zup[:3, :3] @ left_handmarks[0][:3, :3] @ grd_yup2grd_zup[:3, :3].T
    #     print(left_hand_mat[3,:3])    
    #     # z_up, y_left, x_in


    # try:
    #     ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
    #     # print(ypr[:2])
    #     agent._robot.command_joint_state([0., 0.])
    #     agent._robot.command_joint_state(ypr[:2])
    # except:
    #     pass

    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)


    bgr = np.hstack((image_left.numpy()[crop_size_h:, crop_size_w:-crop_size_w],
                     image_right.numpy()[crop_size_h:, crop_size_w:-crop_size_w]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    # rgb = bgr[:,:,:3]
    
    np.copyto(img_array, rgb)

    end = time.time()

print("Closing...")
zed.close()
agent._robot.command_joint_state([0., 0.])