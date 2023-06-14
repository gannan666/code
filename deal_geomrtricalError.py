import os
from glob import glob
import json
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
#import geomeas as gm

dt_dir = '/home/l/download/project1/dt_GE1/'
log_file = "/home/l/download/project1/log_426"
save_file = "/home/l/download/project1/dt_GE2/front_lidar"
dt_infos = {}
delta_infos = {}

def get_box(obj):
    length, width, height = (
        obj["dimension"]["length"],
        obj["dimension"]["width"],
        obj["dimension"]["height"],
    )
    cx = obj["position"]["x"]
    cy = obj["position"]["y"]
    cz = obj["position"]["z"]
    rot_y = obj["rotation"]["yaw"]
    return (cx, cy, cz, length, width, height, rot_y)

def calculate(point_cloud, corners):
    min_z = min(corners[0][2], corners[7][2])
    max_z = max(corners[0][2], corners[7][2])
    min_x = min(corners[0][0], corners[1][0], corners[2][0], corners[3][0])
    max_x = max(corners[0][0], corners[1][0], corners[2][0], corners[3][0])
    min_y = min(corners[0][1], corners[1][1], corners[2][1], corners[3][1])
    max_y = max(corners[0][1], corners[1][1], corners[2][1], corners[3][1])
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_x, min_y, min_z),
                                            max_bound=(max_x, max_y, max_z))
    cropped_point_cloud = point_cloud.crop(aabb).points
    points_num = np.asarray(cropped_point_cloud).shape[0]
    return points_num

def center_to_corners(box):
    pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = box
    half_dim_x, half_dim_y, half_dim_z = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0
    corners = np.array(
        [
            [half_dim_x, half_dim_y, -half_dim_z],
            [half_dim_x, -half_dim_y, -half_dim_z],
            [-half_dim_x, -half_dim_y, -half_dim_z],
            [-half_dim_x, half_dim_y, -half_dim_z],
            [half_dim_x, half_dim_y, half_dim_z],
            [half_dim_x, -half_dim_y, half_dim_z],
            [-half_dim_x, -half_dim_y, half_dim_z],
            [-half_dim_x, half_dim_y, half_dim_z],
        ]
    )
    mm = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, pos_x],
            [np.sin(yaw), np.cos(yaw), 0, pos_y],
            [0, 0, 1.0, pos_z],
            [0, 0, 0, 1.0],
        ]
    )
    corners = (mm[:3, :3] @ corners.T + mm[:3, [3]]).T
    return corners

def corners_to_center(corners):
    x1 = corners[1][0]
    y1 = corners[1][1]
    z1 = corners[1][2]
    x7 = corners[7][0]
    y7 = corners[7][1]
    z7 = corners[7][2]
    x2 = corners[2][0]
    y2 = corners[2][1]
    z2 = corners[2][2]
    x4 = corners[4][0]
    y4 = corners[4][1]
    z4 = corners[4][2]
    P1 = np.array([x1, y1, z1])
    P2 = np.array([x2, y2, z2])
    N1 = np.array([x1-x7, y1-y7, z1-z7])
    N2 = np.array([x2-x4, y2-y4, z2-z4])
    print(gm.Coordinate().calCoordinateFrom2Lines(P1, N1, P2, N2))
    return gm.Coordinate().calCoordinateFrom2Lines(P1, N1, P2, N2)

def update_corners(corners):
    # 检测框在不贴合地面时同时平移下底面
    # TODO 检测框不贴合地面判断
    # corners[0][2]+=0.001
    # corners[1][2]+=0.001
    # corners[2][2]+=0.001
    # corners[3][2]+=0.001
    # 上顶面向上平移
    corners[4][2]+=0.14
    corners[5][2]+=0.14
    corners[6][2]+=0.14
    corners[7][2]+=0.14
    return corners

def update_objects(point_cloud, objects, scale=1.0):
    try_num = 0
    point_delta = 30 # 点云数阈值
    delta = 40
    point_num = []
    if objects is None:
        return []
    for index, obj in enumerate(objects):
        corners = center_to_corners(obj) / scale
        print("corners_to_height: ", corners[7][2] - corners[0][2])       
        point_init = calculate(point_cloud, corners)
        point_num.append(point_init)
        
        while point_delta < delta and try_num < 5:
            corners = update_corners(corners)
            point_new = calculate(point_cloud, corners)
            point_num.append(point_new)
            delta = point_new - point_init
            point_init = point_new
            try_num+=1
        height = corners[7][2] - corners[0][2]
    return height, point_num


def shift(pcd_path, dt_objects=None, scale = 80):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points).reshape(-1, 3)
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points / scale)
    o3d_point_cloud = o3d_point_cloud.paint_uniform_color([1, 0.206, 0])
    if dt_objects is not None:
        # print("len_obj: ", len(dt_objects))
        total_boxes = [get_box(dt_obj) for dt_obj in dt_objects]
        height, point_num = update_objects(o3d_point_cloud, total_boxes, scale=scale)
        print("FINAL point_num and delta height: {} and {}".format(point_num,height))
        delta_infos["{}".format(point_num)] = height
        dt_objects[0]['dimension']['height'] += float(height)
        dt_objects[0]['position']['z'] += float(height)/2
    return dt_objects


def deal_one_timestamp(timestamp):
    timestamp = str(timestamp)
    print("timestamp: ", timestamp)
    if dt_infos.get("pcd_info_dict", None) is None:
        pcd_paths = glob(os.path.join(dt_dir, "pcd_128", "*.pcd"))
        dt_infos["pcd_info_dict"] = {
            os.path.basename(path)[:-4]: path for path in pcd_paths
        }
    if dt_infos.get("json_info_dict", None) is None:
        json_paths = glob(os.path.join(dt_dir, "json_dir", "*.json"))
        dt_infos["json_info_dict"] = {
            os.path.basename(path)[:-5]: path for path in json_paths
        }
    pcd_path = dt_infos["pcd_info_dict"][timestamp]
    dt_json_path = dt_infos["json_info_dict"][timestamp]
    dt_data = json.loads(open(dt_json_path).read())
    dt_objects = dt_data["objects"]

    for obj in dt_objects:
        print("className: ", obj["className"])
        if obj["className"] == "truck" or obj["className"] == "car":
            dt_objects = shift(pcd_path=pcd_path, dt_objects=[obj])
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    with open(save_file + "/{}".format(os.path.basename(dt_json_path)), 'w') as f:
        json.dump(dt_data, f)


if __name__ == "__main__":
    json_path = glob(os.path.join(dt_dir,'*.json'))
    with open(log_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        timestamp = line.strip().split()[3]
        deal_one_timestamp(timestamp)
    # print("DIC: ", delta_infos)
    with open("/home/l/download/delta.json", 'w') as f:
        json.dump(delta_infos, f)