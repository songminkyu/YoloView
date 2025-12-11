# OpenCV based multi camera RTSP stream viewer with multi-threading

This python program open multiple RTSP streams from multiple cameras defined in the camera.json file. When run, it starts with a TKinter GUI dropdown box asking for locations of the cameras you want to view. This location and the associated list of cameras RTSP url is defined in camera.json. It will open an OpenCV window to show all the RTSP streams of the cameras in that location.

Multi-threading is implemented where possible to speed up operation of this program.
