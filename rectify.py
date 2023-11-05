import cv2
import numpy as np
import os
import argparse

def process_image(image_file, output_folder):
    # Try negative in the second loop if not warped
    is_warped = False
    for i in range(2):
        # Load the input image
        image = cv2.imread(image_file)

        # Image upscaling
        # Define the scaling factor (4x upscale)
        scale_factor = 2

        # Calculate the new dimensions
        new_width = image.shape[1] * scale_factor
        new_height = image.shape[0] * scale_factor

        # Resize the image
        image = cv2.resize(image, (new_width, new_height))

        # Apply denoising
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Histogram equalization
        gray = cv2.equalizeHist(gray)

        if i == 1:
            gray = cv2.bitwise_not(gray)

        # Apply Median blur to reduce salt n pepper noise
        blurred = cv2.medianBlur(gray, 7)

        # Apply adaptive thresholding to binarize the image
        thresh = cv2.adaptiveThreshold(
          blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                        mask = np.zeros_like(gray)
                        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

                        # Bitwise AND the mask with the original image to extract the license plate region
                        license_plate = cv2.bitwise_and(image, image, mask=mask)

                        # Convert the license plate region to grayscale
                        plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

                        # Apply Histogram equalization
                        plate_gray = cv2.equalizeHist(plate_gray)

                        # Apply Gaussian blur to the license plate region
                        plate_blurred = cv2.GaussianBlur(plate_gray, (5, 5), 0)

                        # Find contours in the license plate region
                        plate_contours, _ = cv2.findContours(
                          plate_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        # Iterate through the detected contours in the license plate region
                        for plate_contour in plate_contours:
                            # Calculate the convex hull of the license plate contour
                            plate_hull = cv2.convexHull(plate_contour)

                            # Approximate the convex hull to a polygon with fewer vertices (corners)
                            plate_epsilon = 0.07 * cv2.arcLength(plate_hull, True)
                            plate_approx = cv2.approxPolyDP(plate_hull, plate_epsilon, True)

                            # If the polygon has 4 vertices (corners), proceed to warp it
                            if len(plate_approx) == 4:
                                # Convert the polygon to a numpy array of corners
                                plate_corners = np.float32(plate_approx)

                                # Calculate the width and height of the detected license plate
                                plate_width = np.linalg.norm(
                                  plate_corners[0] - plate_corners[1]
                                )
                                plate_height = np.linalg.norm(
                                  plate_corners[1] - plate_corners[2]
                                )

                                # Define the target rectangle dimensions based on the license plate width and height
                                target_plate_width, target_plate_height = int(plate_width), int(
                                  plate_height
                                )

                                # Create the target rectangle corners
                                target_plate_corners = np.float32(
                                  [
                                      [0, 0],
                                      [target_plate_width, 0],
                                      [target_plate_width, target_plate_height],
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
                                  (target_plate_width, target_plate_height),
                                )

                                # Check if the width is shorter than the height
                                if target_plate_width < target_plate_height:
                                  # Rotate the image 90 degrees clockwise
                                  warped_plate = cv2.rotate(
                                      warped_plate, cv2.ROTATE_90_CLOCKWISE
                                  )

                                # Flip the image horizontally
                                warped_plate = cv2.flip(warped_plate, 1)

                                # Save the warped license plate to the output folder
                                output_file = os.path.join(output_folder, f"rectified_{os.path.basename(image_file)}")
                                cv2.imwrite(output_file, warped_plate)
                                is_warped = True

        if is_warped:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Input folder containing images')
    parser.add_argument('-o', '--output_folder', type=str, default='./crop_result', help='Output folder for processed images')
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Iterate through images in the input folder
    for image_file in os.listdir(args.input_folder):
        image_path = os.path.join(args.input_folder, image_file)
        process_image(image_path, args.output_folder)
