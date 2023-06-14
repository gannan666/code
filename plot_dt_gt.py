import os
import glob
import open3d as o3d
import numpy as np
import json
import struct

DT_BOX_COLOR = (0, 0, 255)
GT_BOX_COLOR = (0, 255, 0)

dt_infos = {}
gt_infos = {}

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)


def center_box_to_corners(box):
    # box = box.reshape(-1)
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
    # 这个时候corners还只是平行于坐标轴且以坐标原点为中心来算的.
    mm = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, pos_x],
            [np.sin(yaw), np.cos(yaw), 0, pos_y],
            [0, 0, 1.0, pos_z],
            [0, 0, 0, 1.0],
        ]
    )
    # 然后根据pose,算出真实的,即RX+T
    corners = (mm[:3, :3] @ corners.T + mm[:3, [3]]).T
    return corners


def load_json(path):
    return json.loads(open(path).read())


def parse_obj_get_box(obj):
    """
    haomo's obj
    """
    length, width, height = (
        obj["dimension"]["length"],
        obj["dimension"]["width"],
        obj["dimension"]["height"],
    )
    cx = obj["position"]["x"]
    cy = obj["position"]["y"]
    cz = obj["position"]["z"]
    rot_y = obj["rotation"]["yaw"]
    # className = obj["className"]
    return (cx, cy, cz, length, width, height, rot_y)


def gen_o3d_box3d_lines(objects, label_names, colors, scale=1.0, mode="center"):
    """ """
    assert mode in ["center", "corner"]
    box3d_lines = []
    if objects is None:
        return box3d_lines
    for index, obj in enumerate(objects):
        # compute corners
        if mode == "corner":
            corners = obj / scale  # 8, 3
        else:
            corners = center_box_to_corners(obj) / scale

        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [0, 5],
            [1, 4],
        ]
        if colors is not None:
            color = [colors[index] for i in range(len(lines))]
        else:
            color = [[255, 0, 0] for i in range(len(lines))]  # r g b
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(color)
        box3d_lines.append(line_set)
    return box3d_lines


def plot3d(timestamp, dt_dir, gt_dir, window_name=None):
    if dt_infos.get("pcd_info_dict", None) is None:
        pcd_paths = glob.glob(os.path.join(dt_dir, "pcd", "*.pcd"))
        dt_infos["pcd_info_dict"] = {
            os.path.basename(path)[:-4]: path for path in pcd_paths
        }
    pcd_path = dt_infos["pcd_info_dict"][timestamp]
    print("pcd_path: ", pcd_path)
    if dt_infos.get("json_info_dict", None) is None:
        json_paths = glob.glob(os.path.join(dt_dir, "json", "*.json"))
        dt_infos["json_info_dict"] = {
            os.path.basename(path)[:-5]: path for path in json_paths
        }
    dt_json_path = dt_infos["json_info_dict"][timestamp]
    print("dt: ", dt_json_path)
    dt_data = load_json(dt_json_path)
    dt_objects = dt_data["result"][0]["objects"]
    #dt_objects = dt_data["objects"]
    if len(dt_objects) == 0:
        print("no found dt")
        dt_objects = None
    gt_objects = None
    if gt_infos.get("json_info_dict", None) is None:
        json_paths = glob.glob(os.path.join(dt_dir, "531", "*.json"))
        gt_infos["json_info_dict"] = {
            os.path.basename(path)[:-5]: path for path in json_paths
        }
    gt_json_path = gt_infos["json_info_dict"][timestamp]
    print("gt: ", gt_json_path)
    gt_data = load_json(gt_json_path)
    gt_objects = gt_data["liguo_objects"]
    if len(gt_objects) == 0:
        print("no found gt")
        gt_objects = None  
    
    run3d(
        pcd_path=pcd_path,
        gt_objects=gt_objects,
        dt_objects=dt_objects,
        window_name=window_name,
    )
    return


def run3d(pcd_path, gt_objects=None, dt_objects=None, window_name=None):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points).reshape(-1, 3)
    #  创建画布进行在点云上面开始画东西.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.clear_geometries()
    o3d_point_cloud = o3d.geometry.PointCloud()
    scale = 80
    if window_name is None:
        window_name = os.path.basename(pcd_path)[:-4]
    vis.create_window(window_name)
    vis.get_render_option().background_color = np.asarray(
        [0.0, 0.0, 0.0]
    )  # white background
    vis.get_render_option().point_size = 0.8
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points / scale)
    o3d_point_cloud = o3d_point_cloud.paint_uniform_color([1, 1, 1]) # 把点云颜色统一成白色，否则带有强度信息
    vis.add_geometry(o3d_point_cloud)
    
    '''
    # bin 格式
    example=read_bin_velodyne(pcd_path)
    pcd.points= o3d.open3d.utility.Vector3dVector(example / scale)
    pcd = pcd.paint_uniform_color([1, 1, 1])
    vis.add_geometry(pcd)
    #####################
    '''

    # 添加坐标轴
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=6 / scale, origin=(0, 0, 0)
    )
    vis.add_geometry(FOR1)
    if gt_objects is not None:
        total_boxes = [parse_obj_get_box(gt_obj) for gt_obj in gt_objects]
        length = len(total_boxes)
        label_names = [0] * length
        # Green
        colors = [[0, 255, 0] for _ in label_names]
        line_sets = gen_o3d_box3d_lines(
            total_boxes, label_names, colors, scale=scale, mode="center"
        )
        for line_set in line_sets:
            vis.add_geometry(line_set, False)
    if dt_objects is not None:
        total_boxes = [parse_obj_get_box(dt_obj) for dt_obj in dt_objects]
        length = len(total_boxes)
        label_names = [0] * length
        # Blue
        colors = [[0, 0, 255] for _ in label_names]
        line_sets = gen_o3d_box3d_lines(
            total_boxes, label_names, colors, scale=scale, mode="center"
        )
        for line_set in line_sets:
            vis.add_geometry(line_set, False)

    vis.run()


if __name__ == "__main__":
    dt_dir = "/home/l/下载/531/"
    gt_dir = ""
    json_path = glob.glob(os.path.join(dt_dir, "json", '*.json'))
    for json_i in json_path:
        timestamp = os.path.basename(json_i)[:-5]
        plot3d(timestamp, dt_dir, gt_dir)
