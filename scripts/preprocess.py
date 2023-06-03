import cv2
import os
import glob
import json
import numpy as np
from tqdm import tqdm
import argparse

rotation = {'22053924': cv2.ROTATE_90_CLOCKWISE,
            '22053927': cv2.ROTATE_90_COUNTERCLOCKWISE,
            '22138903': cv2.ROTATE_180,
            '22139915': cv2.ROTATE_180}

# 0: 1500 2048; 1: 3000 4096
scale_mat = {'0': [[0.74325466,  0.,          0.,          0.25293687],
                   [0.,          0.74325466,  0.,         -0.20515952],
                   [0.,          0.,          0.74325466,  0.1476806 ],
                   [0.,          0.,          0.,          1.]],
             '1': [[0.59629434,  0.,          0.,         -0.05127585],
                   [0.,          0.59629434,  0.,         -0.03295311],
                   [0.,          0.,          0.59629434,  0.14789748],
                   [0.,          0.,          0.,          1.]]}


def save_extrinsic(calib, cam_name, extract_folder):
    R = np.array(calib[cam_name]['R']).astype(float).reshape((3, 3))
    T = np.array(calib[cam_name]['T']).astype(float).reshape((3, 1))
    rot = np.eye(3)
    extrinsic = np.zeros((3, 4))
    extrinsic[:3, :3] = rot @ R
    extrinsic[:3, 3:] = rot @ T
    np.save(os.path.join(extract_folder[:-9], f'{cam_name}_extrinsic.npy'), extrinsic)
    # print(extrinsic)


def save_intrinsic_and_img(calib, cam_name, video_path, extract_folder):
    K = np.array(calib[cam_name]['K']).astype(float).reshape((3, 3))
    dist = np.array(calib[cam_name]['distCoeff']).astype(float).reshape((5))
    img_sz = calib[cam_name]['imgSize']
    in_mat = K
    h, w = img_sz[1], img_sz[0]
    new_mat, roi = cv2.getOptimalNewCameraMatrix(in_mat, dist, (w, h), 0, (w, h))
    intrinsic = np.eye(3)

    if h == 4096:
        # h: 4096, w: 3000
        intrinsic[0, 0] = new_mat[0, 0]
        intrinsic[0, 2] = new_mat[0, 2]
        intrinsic[1, 1] = new_mat[1, 1]
        intrinsic[1, 2] = new_mat[1, 2] - 48
        scale = 960 / 3000
        intrinsic[:2] *= scale
        intrinsic[1, 2] -= (640 - 480)
        intrinsic[:2] *= 1024 / 960
    elif h == 3000:
        # h: 3000
        intrinsic[0, 0] = new_mat[0, 0]
        intrinsic[0, 2] = new_mat[0, 2] - 48
        intrinsic[1, 1] = new_mat[1, 1]
        intrinsic[1, 2] = new_mat[1, 2]
        scale = 960 / 3000
        intrinsic[:2] *= scale
        intrinsic[0, 2] -= (640 - 480)
        intrinsic[:2] *= 1024 / 960
    else:
        # h: 1500
        intrinsic[0, 0] = new_mat[0, 0]
        intrinsic[0, 2] = new_mat[0, 2] - 24
        intrinsic[1, 1] = new_mat[1, 1]
        intrinsic[1, 2] = new_mat[1, 2]
        scale = 960 / 1500
        intrinsic[:2] *= scale
        intrinsic[0, 2] -= (640 - 480)
        intrinsic[:2] *= 1024 / 960
    # print(intrinsic)
    np.save(os.path.join(extract_folder[:-9], f'{cam_name}_intrinsic.npy'), intrinsic)
    extract_frames(video_path, extract_folder, in_mat, new_mat, dist)


def extract_frames(video_path, dst_folder, in_mat, new_mat, dist):
    cam_name = dst_folder.split('/')[-1]
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("cannot open the video")
        exit(1)
    index = 0
    print(f'Start processing {video_path.split("/")[-1]}...')
    while True:
        _, frame = video.read()
        if frame is None:
            break
        h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # rotate
        if cam_name in rotation and h == 3000:
            frame = cv2.rotate(frame, rotation[cam_name])
        # crop and resize
        frame = crop_and_resize(frame, in_mat, new_mat, dist)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        save_path = f"{dst_folder}/color{index:05d}.jpg"
        cv2.imwrite(save_path, frame)
        index += 1
    video.release()
    print(f"Saved {index} pics in {dst_folder}")


def crop_and_resize(img, in_mat, new_mat, dist):
    h = img.shape[0]
    if h == 4096:
        img = cv2.undistort(img, in_mat, dist, None, new_mat)
        img = img[48:48 + 4000]
        img = cv2.resize(img, (960, 1280))
        img = img[160:(960 + 160), :]
        img = cv2.resize(img, (1024, 1024))
    elif h == 3000:
        img = cv2.undistort(img, in_mat, dist, None, new_mat)
        img = img[:, 48:48 + 4000]
        img = cv2.resize(img, (1280, 960))
        img = img[:, 160:(960 + 160)]
        img = cv2.resize(img, (1024, 1024))
    else:
        # h = 1500
        img = cv2.undistort(img, in_mat, dist, None, new_mat)
        img = img[:, 24:24 + 2000]
        img = cv2.resize(img, (1280, 960))
        img = img[:, 160:(960 + 160)]
        img = cv2.resize(img, (1024, 1024))
    return img


def process_raw_data(data_dir):
    calib = json.load(open(os.path.join(data_dir, 'calibration_full.json'), 'r'))

    for video_path in glob.glob(os.path.join(data_dir, '*.hevc')):
        extract_folder = video_path[:-5]
        os.makedirs(extract_folder, exist_ok=True)

        cam_name = extract_folder.split('/')[-1]

        # extrinsic
        save_extrinsic(calib, cam_name, extract_folder)

        # save intrinsic and img
        save_intrinsic_and_img(calib, cam_name, video_path, extract_folder)

        os.remove(video_path)


def gen_data(data_dir, output_dir, start_frame, n_frames):
    cam_dict = dict()
    cam_names = sorted([i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))])
    cam_num = len(cam_names)
    data_name = data_dir.split('/')[-1]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    base = start_frame
    n_frames = n_frames
    for j in tqdm(range(n_frames)):
        for i in range(cam_num):
            extrinsic = np.load(f'{data_dir}/{cam_names[i]}_extrinsic.npy')
            pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            pose[:3, :4] = extrinsic
            intr = np.load(f'{data_dir}/{cam_names[i]}_intrinsic.npy')
            intrinsic = np.diag([1, 1, 1, 1]).astype(np.float32)
            intrinsic[:3, :3] = intr
            w2c = pose
            world_mat = intrinsic @ w2c
            world_mat = world_mat.astype(np.float32)
            cam_dict['camera_mat_{}'.format(i + j * cam_num)] = intrinsic
            cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
            cam_dict['world_mat_{}'.format(i + j * cam_num)] = world_mat
            cam_dict['world_mat_inv_{}'.format(i + j * cam_num)] = np.linalg.inv(world_mat)
            cam_dict['fid_{}'.format(i + j * cam_num)] = j

        # center = np.zeros(3)
        # radius = 1.0
        # scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        # scale_mat[:3, 3] = center

        for i in range(cam_num):
            if 'lbn' in data_name or 'yxd' in data_name:
                cam_dict['scale_mat_{}'.format(i + j * cam_num)] = np.array(scale_mat['1'])
                cam_dict['scale_mat_inv_{}'.format(i + j * cam_num)] = np.linalg.inv(np.array(scale_mat['1']))
            else:
                cam_dict['scale_mat_{}'.format(i + j * cam_num)] = np.array(scale_mat['0'])
                cam_dict['scale_mat_inv_{}'.format(i + j * cam_num)] = np.linalg.inv(np.array(scale_mat['0']))

        for i in range(cam_num):
            img = cv2.imread(f'{data_dir}/{cam_names[i]}/color{j + base:05d}.jpg')
            mask = cv2.imread(f'{data_dir}/{cam_names[i]}/mask{j + base:05d}.jpg')
            cv2.imwrite(os.path.join(output_dir, 'image', '{:0>3d}.png'.format(i + j * cam_num)), img)
            cv2.imwrite(os.path.join(output_dir, 'mask', '{:0>3d}.png'.format(i + j * cam_num)), mask)
    np.savez(os.path.join(output_dir, 'cameras_sphere.npz'), **cam_dict)
    print('Process done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/Tensor4D_data/srz-6views')
    parser.add_argument('--output_dir', type=str, default='/data/Tensor4D_data/srz')
    parser.add_argument('--mode', type=str, default='generate', help='process or generate')
    parser.add_argument('--start_frame', type=int, default=19, help='start frame index')
    parser.add_argument('--n_frames', type=int, default=20, help='number frames')

    args = parser.parse_args()
    data_dir = args.data_dir

    if args.mode == 'process':
        process_raw_data(data_dir)
    else:
        output_dir = args.output_dir
        start_frame = args.start_frame
        n_frames = args.n_frames
        gen_data(data_dir, output_dir, start_frame, n_frames)

