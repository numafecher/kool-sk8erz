import cv2

def test_camera():
    # Initialize the default camera (0 is usually the built-in webcam)
    # Change to 1 or 2 if you have external USB cameras connected
    cap = cv2.shape = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the camera. Check your permissions or index.")
        return

    print("Camera successfully connected! Press 'q' on the preview window to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not grabbed successfully, break the loop
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        # Display the resulting live video feed
        cv2.imshow('OpenCV Camera Test', frame)

        # Wait for 1 millisecond and check if 'q' is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing camera feed...")
            break

    # Release the camera hardware resource and destroy all open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
