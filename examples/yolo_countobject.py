import cv2

from ultralytics import solutions

def ObjectCounter_test1(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points
    # region_points = [(20, 400), (1080, 400)]  # For line counting
    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]  # For rectangle region counting
    # region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting

    # Video writer
    video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Init Object Counter
    counter = solutions.ObjectCounter(
        show=True,  # Display the output
        region=region_points,  # Pass region points
        model=model_path,  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
        # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
        # show_in=True,  # Display in counts
        # show_out=True,  # Display out counts
        # line_width=2,  # Adjust the line width for bounding boxes and text display
    )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def count_objects_in_region(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points = [(20, 400), (1080, 400)]
    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


# ObjectCounter_test1('@sample/traffic 2.mp4','models/Yolo11/yolo11s.pt')
# count_objects_in_region("@sample/traffic 2.mp4", "output_video.avi", "models/Yolo11/yolo11s.pt")
# count_specific_classes("@sample/traffic 2.mp4", "output_specific_classes.avi", "models/Yolo11/yolo11s.pt", [0, 2])