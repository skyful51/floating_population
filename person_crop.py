import os
from os import path as osp
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64

data_dir = '01.데이터/1.Training/원천데이터'

for dirpath, dirname, filename in os.walk(data_dir):
    files = os.listdir(dirpath)
    if osp.isdir(osp.join(dirpath, files[0])):
        continue
    else:
        print(dirpath)

for dirpath, dirname, filename in os.walk(data_dir):
    files = os.listdir(dirpath)
    if osp.isdir(osp.join(dirpath, files[0])):
        continue
    else:
        print(dirpath)

    # json_dir = '01.데이터/1.Training/라벨링데이터_1107_add/TL1_do-sa/새벽(0~9)/'
    # video_dir = '01.데이터/1.Training/원천데이터/TS1_do-sa/새벽(0~9)/'
    video_dir = dirpath
    video_list = os.listdir(video_dir)
    json_dir = video_dir.replace('원천데이터', '라벨링데이터')
    print(json_dir)
    json_dir = json_dir.replace('TS', 'TL')
    json_list = os.listdir(json_dir)
    # save_dir = 'person_crop/'
    save_dir = osp.join('person_crop', json_dir.split('/')[-2], json_dir.split('/')[-1])

    if not osp.exists(osp.join('person_crop', json_dir.split('/')[-2])):
        os.mkdir(osp.join('person_crop', json_dir.split('/')[-2]))

    for video in video_list:

        with open(osp.join(json_dir, video.replace('.mp4', '.json')), 'r') as f:
            json_data = json.load(f)

        print(json_data['video']['file_name'])

        # get people info and number of frames from video file
        people = json_data['annotations']
        num_frames = json_data['video']['total_frame']

        # load video file
        print(f'loading video file {video_list}...', end='\r')
        cap = cv2.VideoCapture(osp.join(video_dir, video))
        print(f'loaded video file {video}...')
        cam_id = json_data['video']['file_name'].split('_')[-1].replace('.mp4', '')
        print(f'cam_id: {cam_id}')
        print(f'num of frames: {num_frames}')
        print('-'*50)

        for frame_num in range(num_frames):
            people_in_frame = list(filter(lambda x: x['frame'] == frame_num, people))

            # get new video frame from cap
            _, frame = cap.read()

            for person in people_in_frame:
                x1, y1, x2, y2 = list(map(int, person['bbox']))
                save_name = f'{person["id"]}_{cam_id}_{frame_num}_{person["top_type"]}_{person["top_color"]}_{person["bottom_type"]}_{person["bottom_color"]}'
                print(save_name)
                bbox = frame[y1:y2,x1:x2]

                # save cropped image
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                
                if not osp.exists(osp.join(save_dir, json_data['video']['file_name'].replace('.mp4', ''))):
                    os.mkdir(osp.join(save_dir, json_data['video']['file_name'].replace('.mp4', '')))

                cv2.imwrite(osp.join(save_dir, json_data['video']['file_name'].replace('.mp4', ''), save_name + '.jpg'), bbox)

                # save json file for each cropped image
                labelme_json = dict()
                labelme_json['top_type'] = person['top_type']
                labelme_json['top_color'] = person['top_color']
                labelme_json['bottom_type'] = person['bottom_type']
                labelme_json['bottom_color'] = person['bottom_color']
                labelme_json['version'] = "4.5.12"
                labelme_json['flags'] = {}
                labelme_json['shapes'] = []
                _, img_enc = cv2.imencode('.jpg', bbox)
                img_b64 = base64.b64encode(img_enc).decode('utf-8')
                labelme_json['imagePath'] = save_name + '.jpg'
                labelme_json['imageData'] = img_b64
                labelme_json['imageHeight'] = bbox.shape[0]
                labelme_json['imageWidth'] = bbox.shape[1]

                with open(osp.join(save_dir, json_data['video']['file_name'].replace('.mp4', ''), save_name + '.json'), 'w') as outfile:
                    json.dump(labelme_json, outfile, indent=4)

        print('-'*50)