import cv2
import numpy as np
import os
import argparse


def order_corners(corners):
    # Reshape corners array
    corners_flat = corners.reshape(-1, 2)

    # Calculate centroid of the corners
    centroid = np.mean(corners_flat, axis=0)

    # Calculate angles of each corner with respect to the centroid
    angles = np.arctan2(
        corners_flat[:, 1] - centroid[1], corners_flat[:, 0] - centroid[0]
    )

    # Order corners based on angles
    order = np.argsort(angles)
    sorted_corners = corners_flat[order]

    # Reshape to the original format
    sorted_corners = sorted_corners.reshape(4, 2)

    # Check condition and swap if needed
    if sorted_corners[1][0] < sorted_corners[3][0]:
        sorted_corners[1], sorted_corners[3] = (
            sorted_corners[3].copy(),
            sorted_corners[1].copy(),
        )

    if sorted_corners[1][1] > sorted_corners[2][1]:
        sorted_corners[1], sorted_corners[2] = (
            sorted_corners[2].copy(),
            sorted_corners[1].copy(),
        )

    if sorted_corners[0][1] > sorted_corners[3][1]:
        sorted_corners[0], sorted_corners[3] = (
            sorted_corners[3].copy(),
            sorted_corners[0].copy(),
        )

    return sorted_corners


def process_image(image_file, output_folder):
    # Try negative in the second loop if not warped
    is_warped = False
    for i in range(3):
        # Load the input image
        image = cv2.imread(os.path.join(dir, image_file))

        # Resize the image
        image = cv2.resize(image, (240, 200))

        if i == 2:
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Histogram equalization
        if i == 0 or i == 1:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            gray = clahe.apply(gray)

        if i == 1:
            gray = cv2.bitwise_not(gray)

        if i == 2:
            gray = cv2.equalizeHist(gray)
            gray = cv2.medianBlur(gray, 7)

        # Apply adaptive thresholding to binarize the image
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a copy of the original image
        image_copy = image.copy()

        # Iterate through the detected contours
        for contour in contours:
            # Calculate the contour area
            area = cv2.contourArea(contour)

            # Set a threshold for contour area to filter out small noise
            min_contour_area = 3000

            if area > min_contour_area:
                # Fit a bounding rectangle to the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate the aspect ratio of the bounding rectangle
                aspect_ratio = float(w) / h

                # Set a threshold for the aspect ratio to filter out non-rectangular shapes
                max_aspect_ratio = 3

                if aspect_ratio < max_aspect_ratio:
                    # Check if both width and height are greater than 100
                    if w > 100 and h > 100:
                        # Create a mask for the detected region
                        contour = cv2.convexHull(contour)
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        contour = cv2.approxPolyDP(contour, epsilon, True)

                        mask = np.zeros_like(gray)
                        cv2.drawContours(
                            mask, [contour], -1, (255), thickness=cv2.FILLED
                        )

                        # Bitwise AND the mask with the original image to extract the license plate region
                        license_plate = cv2.bitwise_and(image, image, mask=mask)

                        # Convert the license plate region to grayscale
                        plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

                        # Find contours in the license plate region
                        thresh = cv2.adaptiveThreshold(
                            plate_gray,
                            255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV,
                            11,
                            2,
                        )

                        plate_contours, _ = cv2.findContours(
                            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        # Iterate through the detected contours in the license plate region
                        for plate_contour in plate_contours:
                            # Calculate the contour area
                            area = cv2.contourArea(plate_contour)

                            # Set a threshold for contour area to filter out small noise
                            min_contour_area = 6000

                            if area > min_contour_area:
                                # Fit a bounding rectangle to the contour
                                x, y, w, h = cv2.boundingRect(contour)

                                # Calculate the aspect ratio of the bounding rectangle
                                aspect_ratio = float(w) / h

                                # Set a threshold for the aspect ratio to filter out non-rectangular shapes
                                max_aspect_ratio = 3

                                if aspect_ratio < max_aspect_ratio:
                                    # Check if both width and height are greater than 100
                                    if w > 100 and h > 100:
                                        # Calculate the convex hull of the license plate contour
                                        plate_hull = cv2.convexHull(plate_contour)

                                        # Approximate the convex hull to a polygon with fewer vertices (corners)
                                        plate_epsilon = 0.04 * cv2.arcLength(
                                            plate_hull, True
                                        )
                                        plate_approx = cv2.approxPolyDP(
                                            plate_hull, plate_epsilon, True
                                        )

                                        # If the polygon has 4 vertices (corners), proceed to warp it
                                        if len(plate_approx) == 4:
                                            cv2.drawContours(
                                                image_copy,
                                                [plate_contour],
                                                -1,
                                                (0, 255, 0),
                                                2,
                                            )

                                            # Convert the polygon to a numpy array of corners
                                            plate_corners = np.float32(plate_approx)
                                            plate_corners = order_corners(plate_corners)

                                            # Calculate the width and height of the detected license plate
                                            plate_width = np.linalg.norm(
                                                plate_corners[0] - plate_corners[1]
                                            )
                                            plate_height = np.linalg.norm(
                                                plate_corners[1] - plate_corners[2]
                                            )

                                            # Define the target rectangle dimensions based on the license plate width and height
                                            (
                                                target_plate_width,
                                                target_plate_height,
                                            ) = int(plate_width), int(plate_height)

                                            # Create the target rectangle corners
                                            target_plate_corners = np.float32(
                                                [
                                                    [0, 0],
                                                    [target_plate_width, 0],
                                                    [
                                                        target_plate_width,
                                                        target_plate_height,
                                                    ],
                                                    [0, target_plate_height],
                                                ]
                                            )

                                            # Calculate the perspective transformation matrix for the license plate
                                            plate_M = cv2.getPerspectiveTransform(
                                                plate_corners, target_plate_corners
                                            )

                                            # Warp the license plate to the target rectangle
                                            warped_plate = cv2.warpPerspective(
                                                image,
                                                plate_M,
                                                (
                                                    target_plate_width,
                                                    target_plate_height,
                                                ),
                                            )

                                            # Check if the width is shorter than the height
                                            if target_plate_width < target_plate_height:
                                                # Rotate the image 90 degrees clockwise
                                                warped_plate = cv2.rotate(
                                                    warped_plate,
                                                    cv2.ROTATE_90_CLOCKWISE,
                                                )

                                            # Flip the image horizontally
                                            warped_plate = cv2.flip(warped_plate, 1)

                                            # Save the warped license plate to the output folder
                                            output_file = os.path.join(
                                                output_folder,
                                                f"rectified_{os.path.basename(image_file)}",
                                            )
                                            cv2.imwrite(output_file, warped_plate)
                                            is_warped = True

        if is_warped:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Input folder containing images",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="./crop_result",
        help="Output folder for processed images",
    )
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Iterate through images in the input folder
    for image_file in os.listdir(args.input_folder):
        image_path = os.path.join(args.input_folder, image_file)
        process_image(image_path, args.output_folder)
