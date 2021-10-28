import pyrealsense2 as rs                 
APIpipe = rs.pipeline()

from PIL import Image
import numpy as np
import cv2

import matplotlib.pyplot as plt 
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D


def percentError(pred, gt):
    return np.abs((gt - pred)/pred)*100




def measurement(pred, gt):

    pipe = rs.pipeline()
    cfg = rs.config()
    # INSERT BAG FILE NAME HERE 
    cfg.enable_device_from_file("data/asu_original/12.bag")
    profile = pipe.start(cfg)
    for x in range(50):
        pipe.wait_for_frames()

    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    pipe.stop()
    print("Frames Captured")
    colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)
    aligned_frameset = align.process(frameset)
    aligned_depth_frame = aligned_frameset.get_depth_frame()
    print("DONE")

    # Obtain prediction and GT
    pred = cv2.imread('data/asu_original/output_12.png')
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    gt = cv2.imread('data/asu_original/labels/12.png')
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    
    # Obtain contour of interest mask from prediction
    predOneHot = np.zeros((4,720,1280), dtype = np.uint8)
    predOneHot[1][pred == 1] = 1
    predPitting = predOneHot[1]
    contours, hierarchy = cv2.findContours(predPitting, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursSorted = sorted(contours, key = lambda x: cv2.arcLength(x, True))
    defect = np.zeros_like(predPitting)
    cv2.drawContours(defect, contoursSorted, len(contoursSorted)-1, 255, 1)

    # Obtain X,Y,Z images from depth frame
    pc = rs.pointcloud()
    pointsContainer = pc.calculate(aligned_depth_frame)
    points = np.asarray(pointsContainer.get_vertices())
    points = points.view(np.float32).reshape(points.shape + (-1,))
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    x = x.reshape((720,1280))
    y = y.reshape((720,1280))
    z = z.reshape((720,1280))

    # Calculate point indices list (X,Y,Z)
    pixelIndices = np.where(defect == 255)
    xList = x[pixelIndices]
    yList = y[pixelIndices]
    zList = z[pixelIndices]
    pointsList = np.transpose(np.asarray([xList, yList, zList]))
    pointsList = pointsList[~np.all(pointsList == 0, axis = 1)]
    hull = ConvexHull(pointsList)
    vertices = hull.vertices.tolist() + [hull.vertices[0]]
    area = hull.area


    # GROUND TRUTH AREA
    gtOneHot = np.zeros((4,720,1280), dtype = np.uint8)
    gtOneHot[1][gt == 1] = 1
    gtPitting = gtOneHot[1]
    gtcontours, gthierarchy = cv2.findContours(gtPitting, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gtcontoursSorted= sorted(gtcontours, key = lambda x: cv2.arcLength(x, True))
    gtDefect = np.zeros_like(gtPitting)
    cv2.drawContours(gtDefect, gtcontoursSorted, len(gtcontoursSorted)-1, 255, 1)

    gtPixelIndices = np.where(gtDefect == 255)
    gtxList = x[gtPixelIndices]
    gtyList = y[gtPixelIndices]
    gtzList = z[gtPixelIndices]
    gtpointsList = np.transpose(np.asarray([gtxList, gtyList, gtzList]))
    gtpointsList = gtpointsList[~np.all(gtpointsList == 0, axis = 1)]
    gtHull = ConvexHull(gtpointsList)
    gtvertices = gtHull.vertices.tolist() + [gtHull.vertices[0]]
    gtarea = gtHull.area

    print('AREA (PRED):', area)
    print('AREA (GT):', gtarea)

    return area