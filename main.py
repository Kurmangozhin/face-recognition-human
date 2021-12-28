import argparse
from face_module import face_as_videos
from tqdm import tqdm




if __name__=="__main__":
    # presets
    for i in tqdm(range(10)):
        i = i+1
        face_as_videos(input_dir_image=f'person000{i}', output_path=f'out/000{i}.webm')
    #face_as_videos(input_dir_image = 'person0001', output_path = 'out/0001.webm')
    print('finish')