from IPython import embed
import pickle
import json
import numpy as np
import os
from tqdm import tqdm
import copy
import glob
import open3d as o3d


def get_pcd_path(data):
    if "lidar" in data:
        lidar_dict = {}
        for item in data["lidar"]:
            lidar_dict[item["name"]] = item
            if "center_128_lidar_scan_data" in lidar_dict:
                break
        pcd_path = lidar_dict["center_128_lidar_scan_data"]["oss_path_pcd_txt"]
    elif "point_cloud_Url" in data:
        pcd_path = data["point_cloud_Url"]
    else:
        assert False
    return pcd_path


def parse_obj_get_box(obj):
    """
    haomo's obj
    """
    length, width, height = obj["dimension"]["length"], obj["dimension"]["width"], obj["dimension"]["height"]
    cx, cy, cz = obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]
    rot_y = obj["rotation"]["yaw"]
    return (cx, cy, cz, length, width, height, rot_y)


def parse_obj_get_box_yayun(obj):
    """
    haomo's yayun obj
    """
    length, width, height = obj["lidar_coor"]["dimension"]["length"], obj["lidar_coor"]["dimension"]["width"], obj["lidar_coor"]["dimension"]["height"]
    cx, cy, cz = obj["lidar_coor"]["position"]["x"], obj["lidar_coor"]["position"]["y"], obj["lidar_coor"]["position"]["z"]
    rot_y = obj["lidar_coor"]["rotation"]["yaw"]
    return (cx, cy, cz, length, width, height, rot_y)


def parse_json_haomo(box_path):
    ret = []
    data = json.loads(open(box_path).read())
    objs = data["objects"]
    for obj in objs:
        box = parse_obj_get_box(obj)
        ret.append(box)
    return np.array(ret).reshape(-1, 7)


def center_box_to_corners(box):
    pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = box
    half_dim_x, half_dim_y, half_dim_z = dim_x/2.0, dim_y/2.0, dim_z/2.0
    corners = np.array([[half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, half_dim_y, half_dim_z],
                        [half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, half_dim_y, half_dim_z]])
    # 这个时候corners还只是平行于坐标轴且以坐标原点为中心来算的.
    transform_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, pos_x],
        [np.sin(yaw), np.cos(yaw), 0, pos_y],
        [0, 0, 1.0, pos_z],
        [0, 0, 0, 1.0],
    ])
    # 然后根据pose,算出真实的,即RX+T
    corners = (transform_matrix[:3, :3] @
               corners.T + transform_matrix[:3, [3]]).T
    return corners


def gen_o3d_box3d_lines(objects, label_names, colors, scale=3.0, mode="center"):
    """

    """
    assert mode in ["center", "corner"]
    box3d_lines = []
    box3d_dirs = []
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
                [1, 4]
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


def get_score(obj):
    score = obj["score"]
    return score


def     run(pcd_path, json_path, gt_path, window_name):
    """
    pcd_path: 点云的路径.
    box_path: 
    里面的shape是(m, n, 8, 3) 的shape
    m代表目标的个数, n代表这个目标的框数.
    """
    print("run!!")
    # 读取点云路径.
    pcd = o3d.io.read_point_cloud(pcd_path)
    # points = np.fromfile(pcd_path, "float32").reshape(-1, 4)[:, :3]
    points = np.asarray(pcd.points).reshape(-1, 3)

    #  创建画布进行在点云上面开始画东西.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.clear_geometries()
    o3d_point_cloud = o3d.geometry.PointCloud()
    intensity = None
    label = None
    scale = 80
    vis.create_window(window_name)
    vis.get_render_option().background_color = np.asarray(
        [0.0, 0.0, 0.0])  # white background
    vis.get_render_option().point_size = 0.2
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points / scale)
    o3d_point_cloud = o3d_point_cloud.paint_uniform_color([1, 1, 1])
    vis.add_geometry(o3d_point_cloud)

    # 添加坐标轴
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=8/scale, origin=(0, 0, 0))
    vis.add_geometry(FOR1)
    # mesh_sphere = o3d.geometry.TriangleMesh.create_cylinder(radius=4.0, height=1)
    # vis.add_geometry(mesh_sphere)

    json_data = json.loads(open(json_path).read())
    objects = json_data["objects"]
    # print(objects)
    gt_data = json.loads(open(gt_path).read())
    gt_objects = gt_data["objects"] 
    # liguo_objects = json_data["objects"]  # liguo_objects
    # objects_scores = [get_score(obj) for obj in objects]
    # liguo_objects_scores = [get_score(obj) for obj in liguo_objects]

    dismatch_scores=[]
    y = []
    
    # TODO 绘制liguo_objects与objects的差集: green
    
    # for obj in liguo_objects:
    #         if obj["score"] not in objects_scores and obj["score"]>0.6:
    #             y.append(obj["position"]["y"])
    #             dismatch_scores.append(obj["score"])
    #             total_boxes = [parse_obj_get_box(obj)] # total_boxes = [parse_obj_get_box(obj) for obj in objects]
    #             length = len(total_boxes)
    #             label_names = [0] * length
    #             colors = [[0, 255, 0] for _ in label_names] # r g b
    #             line_sets = gen_o3d_box3d_lines(
    #             total_boxes, label_names, colors, scale=scale, mode="center")
    #             for line_set in line_sets:
    #                 vis.add_geometry(line_set, False)
    #         else:
    #             continue
    # print("scores: ",dismatch_scores)
    # print("y: ", y)
    

    # TODO 绘制liguo_objects
    
    # total_boxes_li = [parse_obj_get_box(obj) for obj in liguo_objects]
    # length_li = len(total_boxes_li)
    # label_names_li = [0] * length_li
    # colors_li = [[0, 0, 255] for _ in label_names_li]
    # line_sets_li = gen_o3d_box3d_lines(
    #     total_boxes_li, label_names_li, colors_li, scale=scale, mode="center")
    # for line_set in line_sets_li:
    #     vis.add_geometry(line_set, False)


    # TODO 绘制objects:r
    
    total_boxes_obj = [parse_obj_get_box(obj) for obj in objects]
    length_obj = len(total_boxes_obj)
    label_names_obj = [0] * length_obj
    colors_obj = [[255, 0, 0] for _ in label_names_obj]
    line_sets_obj = gen_o3d_box3d_lines(
        total_boxes_obj, label_names_obj, colors_obj, scale=scale, mode="center")
    for line_set in line_sets_obj:
        vis.add_geometry(line_set, False)


    # TODO 绘制gt_objects:g
    
    total_boxes_gt = [parse_obj_get_box(obj) for obj in gt_objects]
    length_gt = len(total_boxes_gt)
    label_names_gt = [0] * length_gt
    colors_gt = [[0, 255, 0] for _ in label_names_gt]
    line_sets_gt = gen_o3d_box3d_lines(
        total_boxes_gt, label_names_gt, colors_gt, scale=scale, mode="center")
    for line_set in line_sets_gt:
        vis.add_geometry(line_set, False)
   
    
    # TODO 绘制各个视角下的yayun_objects: blue
    
    # yayun_objects_rl = json_data["yayun_objects"]["rear_left_camera"]["objects"]
    # yayun_objects_fl = json_data["yayun_objects"]["front_left_camera"]["objects"]
    # yayun_objects_rr = json_data["yayun_objects"]["rear_right_camera"]["objects"]
    # yayun_objects_fr = json_data["yayun_objects"]["front_right_camera"]["objects"]
    # yayun_objects_rm = json_data["yayun_objects"]["rear_middle_camera"]["objects"]
    # yayun_objects_fm = json_data["yayun_objects"]["front_middle_camera"]["objects"]
    # # rl 
    # total_boxes_rl = [parse_obj_get_box_yayun(obj) for obj in yayun_objects_rl]
    # length_rl = len(total_boxes_rl)
    # label_names_rl = [0] * length_rl
    # colors_rl = [[0, 0, 255] for _ in label_names_rl]
    # line_sets_rl = gen_o3d_box3d_lines(
    #     total_boxes_rl, label_names_rl, colors_rl, scale=scale, mode="center")
    # for line_set in line_sets_rl:
    #     vis.add_geometry(line_set, False)
    # # fl 
    # total_boxes_fl = [parse_obj_get_box_yayun(obj) for obj in yayun_objects_fl]
    # length_fl = len(total_boxes_fl)
    # label_names_fl = [0] * length_fl
    # colors_fl = [[0, 0, 255] for _ in label_names_fl]
    # line_sets_fl = gen_o3d_box3d_lines(
    #     total_boxes_fl, label_names_fl, colors_fl, scale=scale, mode="center")
    # for line_set in line_sets_fl:
    #     vis.add_geometry(line_set, False)
    # # rr 
    # total_boxes_rr = [parse_obj_get_box_yayun(obj) for obj in yayun_objects_rr]
    # length_rr = len(total_boxes_rr)
    # label_names_rr = [0] * length_rr
    # colors_rr = [[0, 0, 255] for _ in label_names_rr]
    # line_sets_rr = gen_o3d_box3d_lines(
    #     total_boxes_rr, label_names_rr, colors_rr, scale=scale, mode="center")
    # for line_set in line_sets_rr:
    #     vis.add_geometry(line_set, False)
    # # fr 
    # total_boxes_fr = [parse_obj_get_box_yayun(obj) for obj in yayun_objects_fr]
    # length_fr = len(total_boxes_fr)
    # label_names_fr = [0] * length_fr
    # colors_fr = [[0, 0, 255] for _ in label_names_fr]
    # line_sets_fr = gen_o3d_box3d_lines(
    #     total_boxes_fr, label_names_fr, colors_fr, scale=scale, mode="center")
    # for line_set in line_sets_fr:
    #     vis.add_geometry(line_set, False)
    # # rm 
    # total_boxes_rm = [parse_obj_get_box_yayun(obj) for obj in yayun_objects_rm]
    # length_rm = len(total_boxes_rm)
    # label_names_rm = [0] * length_rm
    # colors_rm = [[0, 0, 255] for _ in label_names_rm]
    # line_sets_rm = gen_o3d_box3d_lines(
    #     total_boxes_rm, label_names_rm, colors_rm, scale=scale, mode="center")
    # for line_set in line_sets_rm:
    #     vis.add_geometry(line_set, False)
    # # fm 
    # total_boxes_fm = [parse_obj_get_box_yayun(obj) for obj in yayun_objects_fm]
    # length_fm = len(total_boxes_fm)
    # label_names_fm = [0] * length_fm
    # colors_fm = [[0, 0, 255] for _ in label_names_fm]
    # line_sets_fm = gen_o3d_box3d_lines(
    #     total_boxes_fm, label_names_fm, colors_fm, scale=scale, mode="center")
    # for line_set in line_sets_fm:
    #     vis.add_geometry(line_set, False)

    vis.run()


pcd_paths = glob.glob("/home/l/download/pcd/1667787770100328.pcd")
json_paths = glob.glob("/home/l/download/card2/*.json")
gt_paths  =glob.glob("/home/l/download/gt_dir/*.json")
pcd_paths = sorted(pcd_paths)
json_paths = sorted(json_paths)
gt_paths = sorted(gt_paths)
#print(json_paths)
print(pcd_paths)
# print(gt_paths)

start = 0
for pcd_path, json_path, gt_path in zip(pcd_paths, json_paths, gt_paths):
    timestamp = os.path.basename(pcd_path)[:-4]
    print(pcd_path)
    print(json_path)
    print(gt_path)
    run(pcd_path, json_path, '/home/l/download/gt_dir/' + os.path.basename(json_path), str(start))
    #run(pcd_path, json_path, gt_path, str(start))
    start += 1


# for pcd_path, json_path in zip(pcd_paths, json_paths):
#     timestamp = os.path.basename(pcd_path)[:-4]
#     print(pcd_path)
#     print(json_path)
#     # run(pcd_path, json_path, os.path.basename(json_path))
#     run(pcd_path, json_path, str(start))
#     start += 1