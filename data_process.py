import pickle

import numpy as np
import argparse
#import open3d as o3d
import os
import sys
# import termios
import h5py
import glob

import torch

from similarity import KDTreeSimilar
from network import get_model

scantnetlabel = {0:'bathtub', 1:'bed', 2:'bookshelf', 3:'cabinet', 4:'chair', 5:'lamp', 6:'monitor',7:'plant', 8:'sofa', 9:'table'}

def load_data_h5py_scannet10(partition, dataroot, start=0):
    """
    Input:
      partition - train/test
    Return:
      data,label arrays
    """


    data_path = "./static/scannet_data.npy"
    label_path = "./static/scannet_label.npy"
    if os.path.exists(data_path) and os.path.exists(label_path):
      return (np.load(data_path), np.load(label_path)) if not start else \
          (np.load(data_path)[start:], np.load(label_path)[start:])

    if not os.path.exists("./static"):
      os.mkdir("./static")
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(dataroot, '%s_*.h5' % partition))):
        #if partition and dataroot:
        print("Ethan: ",dataroot)
        f = h5py.File(h5_name, 'r')
        data = f['data'][:]
        label = f['label'][:]
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_data = np.array(all_data).astype('float32')
    all_label = np.array(all_label).astype('int64')
    np.save(data_path, all_data),np.save(label_path, all_label)
    return all_data[start:], all_label[start:]

# def press_any_key_exit():
#   # 获取标准输入的描述符
#   fd = sys.stdin.fileno()
#   # 获取标准输入(终端)的设置
#   old_ttyinfo = termios.tcgetattr(fd)
#   # 配置终端
#   new_ttyinfo = old_ttyinfo[:]
#   # 使用非规范模式(索引3是c_lflag 也就是本地模式)
#   new_ttyinfo[3] &= ~termios.ICANON
#   # 关闭回显(输入不会被显示)
#   new_ttyinfo[3] &= ~termios.ECHO
#   # 输出信息
#   #sys.stdout.write(msg)
#   #sys.stdout.flush()
#   # 使设置生效
#   termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
#   # 从终端读取
#   os.read(fd, 7)
#   # 还原终端设置
#   termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)



def genShapeNetFeatures(net):

    #指定shapeNet数据集路径
    shapeNetPath = r"/data/zhangguozhu/my_code/PointDA_data/shapenet"

    for cls in scantnetlabel:
        fetures_path = f"./static/features/{scantnetlabel[cls]}_features.npy"
        features_path_path = f"./static/features/{scantnetlabel[cls]}_features_path.npy"
        if not os.path.exists(fetures_path):
            shapeNetDataPath = os.path.join(shapeNetPath, scantnetlabel[cls], "train")
            all_features = []
            all_features_path = []
            filelist = os.listdir(shapeNetDataPath)
            for i in filelist:
                print(i)
                data = np.load(os.path.join(shapeNetDataPath, i))
                # print(data.shape)
                # print(type(data))
                net_input = torch.transpose(torch.from_numpy(data).float(), 0, 1)
                #v1通过repeat扩展维度  net_input = net_input.repeat([2,1])
                #改为将rgb都设置为0
                zero_rgb = torch.zeros(net_input.shape)
                net_input = torch.cat((net_input, zero_rgb))
                # print(net_input)
                # print(net_input.shape)
                abs_data_path = os.path.abspath(os.path.join(shapeNetDataPath, i))
                print(abs_data_path)
                feature = net(net_input[None, ...])
                print(feature.view(-1).shape)
                all_features.append(feature.view(-1).detach().numpy()), all_features_path.append(abs_data_path)
            np.save(fetures_path, np.array(all_features)), np.save(features_path_path, np.array(all_features_path))



def genKDTrees():
    features_dir_path = "./static/features"
    kdtree_path = './static/kd_trees.pickle'
    global kdtrees
    if os.path.exists(kdtree_path):
      with open(kdtree_path, 'rb') as f:
        kdtrees = pickle.load(f)
        print(kdtrees)
        print(type(kdtrees))
      return
    for cls in scantnetlabel:
          features_file, features_path_file = sorted(glob.glob(os.path.join(features_dir_path, f"{scantnetlabel[cls]}_features*.npy")))
          print(features_file, features_path_file)
          features = np.load(features_file)
          features_path = np.load(features_path_file)
          kdtrees[cls] = KDTreeSimilar(features, features_path)
    with open(kdtree_path, 'wb') as f:
      pickle.dump(kdtrees, f)

      

if __name__=="__main__":
    #加载网络模型
    net = get_model(50, True)
    parm_path = "../log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth"
    net.load_state_dict(torch.load(parm_path)['model_state_dict'])
    #开启评估模式
    net.eval()
    #生成shapeNet所有点云数据的特征
    genShapeNetFeatures(net)

    #根据生成的特征构建不同类别的KDTree
    kdtrees = {}
    genKDTrees()

    
    #保存数据集的处理记录
    start = 0
    #如果需要修改指定start的位置  请先将recode_num.npy文件删除
    record_path = "./static/record_num.npy"
    if os.path.exists(record_path):
        start = np.load(record_path)
        print(start)

    scanNetPath = r"/data/zhangguozhu/my_code/PointDA_data/scannet"
    scantdata, scantlabel = load_data_h5py_scannet10('train', scanNetPath, start)
    #print(scantdata, scantlabel)
    for i in range(scantdata.shape[0]):
        scantdata_xyz = scantdata[i][:, :3]
        #可视化操作
        # txt_data = np.savetxt('./tmp/scene1.txt', scantdata_xyz)
        # print(scantdata_xyz.shape)
        # pcd = o3d.io.read_point_cloud('./tmp/scene1.txt', format='xyz')
        # print(pcd)
        # print(type(pcd))
        # o3d.visualization.draw_geometries([pcd], window_name=str(scantnetlabel[scantlabel[i]]), width=1200, height=600, point_show_normal=True) # 可视化点云
        #print("scant label :", scantlabel[i])

        #获取当前点云特征
        net_input = torch.transpose(torch.from_numpy(scantdata_xyz).float(), 0, 1)
        #net_input = net_input.repeat([2, 1])
        #v1通过repeat扩展维度  net_input = net_input.repeat([2,1])
        #改为将rgb都设置为0
        zero_rgb = torch.zeros(net_input.shape)
        net_input = torch.cat((net_input, zero_rgb))
        #print(net_input)
        feature = net(net_input[None, ...])
        #print(feature.view(-1).shape)
        #print(feature.view(-1))
        #从对应类别的kdtree中查找最相近的前k个
        k_similarity_path =  kdtrees[scantlabel[i]].predict((feature.view(1, -1).detach().numpy()), k=1)[0]
        #print(i, k_similarity_path[0])        
        flag = True
        while(flag):
            for p in k_similarity_path:
                data = np.load(p)
                #print(data)
                #print(type(data))

                #可视化操作
                # txt_data = np.savetxt('./tmp/scene2.txt', data)
                # pcd = o3d.io.read_point_cloud('./tmp/scene2.txt', format='xyz')
                # print(pcd)
                # print(type(pcd))
                # o3d.visualization.draw_geometries([pcd], window_name="Point 3D", width=1200, height=600,
                #                                   point_show_normal=True)


                #人工进行在最相近得前k个中进行挑选
                # c= input("是否继续查看最相似的k个点云?  请输入 y/n: ")
                c = 'n'
                if c== 'n':
                    flag = False
                    with open("./static/result.txt", "a", encoding='utf-8') as f:
                        f.write(f"{start} {os.path.basename(p)}\n")
                    break

        #人工处理
        # choice = input("是否继续进行数据处理?  请输入 y/n: ")
        choice = 'y'
        start += 1
        if choice == 'n':
          np.save(record_path, start)
          break
    np.save(record_path, start)
    print("processing  complete...")



    """
    
    parse = argparse.ArgumentParser(description="point cloud visualization")
    #完整点云数据  shapeNet .npy数据文件的父级目录
    parse.add_argument('--shapeNetPath', type=str, default='./', help='root path')
    #残缺点云数据  scanNet .h5数据文件的父级目录
    parse.add_argument('--scanNetPath', type=str, help='root path')
    #指定单个文件的具体地址
    parse.add_argument('--filePath', type=str, help='root path')
    parse.add_argument('--normal', type=bool, default=False,help='root path')
    args = parse.parse_args()
    np.set_printoptions(suppress=True)
    # 作用是取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
    if args.scanNetPath:
      print("Ethan : ", args.scanNetPath)
      scantdata, scantlabel = load_data_h5py_scannet10('train', args.scanNetPath)
      for i in range(scantdata.shape[0]):
        txt_data = np.savetxt('./tmp/scene1.txt', scantdata[i])
        # press_any_key_exit()
        #pcd = o3d.io.read_point_cloud('scene1.txt', format='xyzrgb')
        pcd = o3d.io.read_point_cloud('./tmp/scene1.txt', format='xyz')
        # 此处因为npy里面正好是 x y z r g b的数据排列形式，所以format='xyzrgb'
        print(pcd)
        print(type(pcd))
        if args.normal:
          pcd.voxel_down_sample(voxel_size=0.0564)
          pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("scant label :", scantlabel[i])
        o3d.visualization.draw_geometries([pcd], window_name=str(scantnetlabel[scantlabel[i]]), width=1200, height=600, point_show_normal=True) # 可视化点云
        choice = input("是否继续查看?  请输入 y/n: ")
        if choice == 'n':
          break
    else:
      if args.filePath :
        #press_any_key_exit()
        data = np.load(args.file)
        print(data)
        print(type(data))
        txt_data = np.savetxt('./tmp/scene1.txt', data)
        #pcd = o3d.io.read_point_cloud('scene1.txt', format='xyzrgb')
        pcd = o3d.io.read_point_cloud('./tmp/scene1.txt', format='xyz')
        # 此处因为npy里面正好是 x y z r g b的数据排列形式，所以format='xyzrgb'
        print(pcd)
        print(type(pcd))
        if args.normal:
          pcd.voxel_down_sample(voxel_size=0.0564)
          pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d.visualization.draw_geometries([pcd], window_name="Point 3D",width=1200, height=600, point_show_normal=True) # 可视化点云
        #o3d.visualization.draw([pcd],width=1200, height=600) # 可视化点云
        #o3d.visualization.draw_geometries([pcd], width=1200, height=600, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True) # 可视化点云
        #o3d.visualization.draw_geometries([pcd], width=1200, height=600, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True) # 可视化点云
      else:
        if args.shapeNetPath:
          filelist = os.listdir(args.shapeNetPath)
          for i in filelist:
            print(i)
            # press_any_key_exit()
            data = np.load(os.path.join(args.shapeNetPath,i))
            print(data)
            print(type(data))
            txt_data = np.savetxt('./tmp/scene1.txt', data)
            #pcd = o3d.io.read_point_cloud('scene1.txt', format='xyzrgb')
            pcd = o3d.io.read_point_cloud('./tmp/scene1.txt', format='xyz')
            # 此处因为npy里面正好是 x y z r g b的数据排列形式，所以format='xyzrgb'
            print(pcd)
            print(type(pcd))
            if args.normal:
              pcd.voxel_down_sample(voxel_size=0.0564)
              pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            o3d.visualization.draw_geometries([pcd], window_name="Point 3D",width=1200, height=600, point_show_normal=True) # 可视化点云
            choice = input("是否继续查看?  请输入 y/n: ")
            if choice == 'n':
              break
            #o3d.visualization.draw([pcd],width=1200, height=600) # 可视化点云
            #o3d.visualization.draw_geometries([pcd], width=1200, height=600, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True) # 可视化点云
            #o3d.visualization.draw_geometries([pcd], width=1200, height=600, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True) # 可视化点云
        else:
          print("inpur file and path")
          
    """