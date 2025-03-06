import argparse
import sys
import time
import numpy as np
import cv2

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient

from gradio_client import Client, handle_file
import re
import base64


def image_to_opencv(image):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = ".png"
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = ".jpg"

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into an RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    return img, extension


def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = (
            "Robot is estopped. Please use an external E-Stop client, such as the"
            " estop SDK example, to configure E-Stop."
        )
        robot.logger.error(error_message)
        raise Exception(error_message)


def arm_object_grasp(config):
    """A simple example of using the Boston Dynamics API to command Spot's arm with RoboPoint VLM."""

    client = Client(config.gradio_app_hostname)
    result = client.predict(api_name="/load_demo_refresh_model_list")

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk("ArmObjectGraspClient")
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # forcefully take the lease
    lease_client.take()

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=False):

        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Take a picture with a camera
        robot.logger.info("Getting an image from: %s", config.image_source)

        # construct the iamge request
        request = [
            build_image_request(
                config.image_source + "_fisheye_image",
                quality_percent=100,
                resize_ratio=1,
                pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
            ),
            build_image_request(
                config.image_source + "_depth_in_visual_frame",
                quality_percent=100,
                resize_ratio=1,
                pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
            ),
        ]

        # images are in the same order as the request
        image = image_client.get_image(request)[0]
        image_depth = image_client.get_image(request)[1]

        img, _ = image_to_opencv(image)

        # gradio needs a image file or url to process the image
        cv2.imwrite("grasp.jpg", img)

        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_depth.shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_depth.shot.image.rows, image_depth.shot.image.cols)

        # set the image and request to the gradio inference api
        result = client.predict(
            text=config.request, image=handle_file("grasp.jpg"), image_process_mode="Pad", api_name="/add_text_1"
        )

        # run the inference
        result = client.predict(
            model_selector="robopoint-v1-vicuna-v1.5-13b",
            temperature=1,
            top_p=0.7,
            max_new_tokens=512,
            api_name="/http_bot_2",
        )

        # Extract the base64 image string
        img_tag = result[0][1]
        base64_str = re.search(r'data:image/jpeg;base64,(.*?)"', img_tag).group(1)

        # Decode the base64 string to get the image
        image_data = base64.b64decode(base64_str)

        # Save the camera image to a file (optional)
        with open("input.jpg", "wb") as f:
            f.write(image_data)

        # Extract the coordinates
        coordinates_str = re.search(r"\[(.*?)\]▌", img_tag).group(1)
        coordinates = eval(coordinates_str)

        print("Coordinates:", coordinates)
        # take the first 2d action point as an example for the grasp
        x = coordinates[0][0] * image.shot.image.cols
        y = coordinates[0][1] * image.shot.image.rows
        robot.logger.info(f"Picking object at image location ({x}, {y})")
        
        #create the pick proto vector
        pick_vec = geometry_pb2.Vec2(x=x, y=y)

        # wait for user input to verify the grasp
        input("Attention: Press Enter to continue and start the grasp...")
        robot.logger.info("Starting grasp...")

        # Build the proto and 2D to 3D projection using the API call
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole,
        )

        # Optionally add a grasp constraint.
        add_grasp_constraint(config, grasp, robot_state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request
            )

            print(f"Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}")

            if (
                response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
                or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
            ):
                break

            time.sleep(0.25)

        time.sleep(2)
        robot.logger.info("Finished grasp.")

        # set the arm to carry
        carry_cmd = RobotCommandBuilder.arm_carry_command()
        command_client.robot_command(carry_cmd)


def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if config.force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.
            # That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("-i", "--image-source", help="Get image from source", default="frontleft")
    parser.add_argument(
        "-t",
        "--force-top-down-grasp",
        help="Force the robot to use a top-down grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--force-horizontal-grasp",
        help="Force the robot to use a horizontal grasp (vector_alignment demo)",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--force-45-angle-grasp",
        help="Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--force-squeeze-grasp", help="Force the robot to use a squeeze grasp", action="store_true"
    )
    parser.add_argument("-l", "--request", help="Request", type=str)
    parser.add_argument("-g", "--gradio-app-hostname", help="Gradio app hostname", type=str, required=True)

    options = parser.parse_args()

    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1

    if num > 1:
        print("Error: cannot force more than one type of grasp.  Choose only one.")
        sys.exit(1)

    try:
        arm_object_grasp(options)
        return True
    except Exception:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
