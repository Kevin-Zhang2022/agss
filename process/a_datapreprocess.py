import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from natsort import natsorted
import matplotlib


from SimpleAuditory.model import cochlea, auditorynerve, stellate, harmolearn
import scipy.ndimage
import numpy as np



def resize(image, output_shape=(224, 224)):
    # Assuming 'image' is a 2D numpy array
    resized_image = scipy.ndimage.zoom(image, (output_shape[0] / image.shape[0], output_shape[1] / image.shape[1]), order=1)
    return resized_image


def crop_pad(y, target_length):
    current_length = len(y)
    if current_length < target_length:
        # 如果音频长度不足目标长度，则在末尾填充零
        padding = np.zeros(target_length - current_length)
        y = np.concatenate((y, padding))
    elif current_length > target_length:
        # 如果音频长度超过目标长度，则截断
        y = y[:target_length]
    return y

def save_cocgram(wav_path,coc,an):
    y, sr = librosa.load(wav_path, sr=44100)
    if wav_path.split('/')[2] == 'us8k':
        y = crop_pad(y, 44100*4)
    if wav_path.split('/')[2] == 'esc10':
        y = crop_pad(y, 44100*5)

    y = (y-y.mean())/(y.std()+1e-5)
    y = coc(y)
    y = resize(y, output_shape=(224, 224))
    y = an(y)
    # plt.imshow(y, aspect='auto')
    # plt.show()

    # S = resize(S, output_shape=(224, 224))
    # S = (S-S.min())/(S.max()-S.min())
    # S = np.expand_dims(S, axis=0)

    wave_path_img = wav_path.replace('audio', 'img/coc_out').replace('.wav', '.jpg')
    wav_dir_img = os.path.dirname(wave_path_img)
    os.makedirs(wav_dir_img, exist_ok=True)

    wave_path_npy = wav_path.replace('audio', 'npy/coc_out').replace('.wav', '.npy')
    wav_dir_npy = os.path.dirname(wave_path_npy)
    os.makedirs(wav_dir_npy, exist_ok=True)

    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(y, aspect='auto')
    plt.xlabel('Time frame')  # win = (2048/22.05k) ~9.3ms  512 hop 2.3ms
    plt.ylabel('Frequency Channel')
    plt.title(wav_dir_img.split('/')[-1])
    plt.savefig(wave_path_img)
    plt.close('all')

    np.save(wave_path_npy, y)


def save_stella(file_path, stella):
    y = np.load(file_path)
    y = stella(y)

    file_path_img = file_path.replace('npy/coc_out', 'img/stella_c').replace('.npy', '.jpg')
    file_dir_img = os.path.dirname(file_path_img)
    os.makedirs(file_dir_img, exist_ok=True)

    file_path_npy = file_path.replace('npy/coc_out', 'npy/stella_c')
    file_dir_npy = os.path.dirname(file_path_npy)
    os.makedirs(file_dir_npy, exist_ok=True)

    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(y, aspect='auto')
    plt.xlabel('Time frame')  # win = (2048/22.05k) ~9.3ms  512 hop 2.3ms
    plt.ylabel('Frequency Channel')
    plt.title(file_dir_img.split('/')[-1])
    plt.savefig(file_path_img)
    plt.close('all')

    np.save(file_path_npy, y)





def coc_process(dataset_list=('esc10', 'us8k')):
    coc = cochlea(channels=224, window=440)
    an = auditorynerve(uth=0.05)
    dataset_list = ['esc10', 'us8k']
    for dataset in dataset_list:
        src_dir = '../data/'+dataset+'/audio'
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = natsorted(dirs)
            files = natsorted(files)
            for i, file in enumerate(files):
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file).replace('\\', '/')
                    save_cocgram(wav_path, coc, an)
                    print(wav_path, '', i)
                    # print(f'Saved spectrogram for {wav_path} to {tar_dir}')
    print('end')


def stella_process(dataset_list=('esc10', 'us8k')):
    stella = stellate(channels=224, thr=0.5, sigma=3)
    for dataset in dataset_list:
        src_dir = '../data/'+dataset+'/npy/coc_out'
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = natsorted(dirs)
            files = natsorted(files)
            for i, file in enumerate(files):
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file).replace('\\', '/')
                    save_stella(file_path, stella)
                    print(file_path, '', i)
    print('end')


def save_general(file_path, process, src_kw, tar_kw):
    y = np.load(file_path)
    y = process(y)

    file_path_img = file_path.replace(f'npy/{src_kw}', f'img/{tar_kw}').replace('.npy', '.jpg')
    file_dir_img = os.path.dirname(file_path_img)
    os.makedirs(file_dir_img, exist_ok=True)

    file_path_npy = file_path.replace(f'npy/{src_kw}', f'npy/{tar_kw}')
    file_dir_npy = os.path.dirname(file_path_npy)
    os.makedirs(file_dir_npy, exist_ok=True)

    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(y, aspect='auto')
    plt.xlabel('Time frame')  # win = (2048/22.05k) ~9.3ms  512 hop 2.3ms
    plt.ylabel('Frequency Channel')
    plt.title(file_dir_img.split('/')[-1])
    plt.savefig(file_path_img)
    plt.close('all')

    np.save(file_path_npy, y)


def process_general(process, tar_kw, src_kw='coc_out', dataset_list=['esc10', 'us8k']):
    for dataset in dataset_list:
        src_dir = '../data/'+dataset+f'/npy/{src_kw}'
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = natsorted(dirs)
            files = natsorted(files)
            for i, file in enumerate(files):
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file).replace('\\', '/')
                    save_general(file_path, process, src_kw=src_kw, tar_kw=tar_kw)
                    print(file_path, '', i)
    print('end')


if __name__ == "__main__":
    matplotlib.use('Agg')
    # matplotlib.use('TkAgg')
    dataset_list = ['us8k']
    stella = stellate(channels=224, thr=0.5, sigma=7.5)  # 'stella_c'
    hl = harmolearn(channels=224, window=8, w_range=(0, 1.5))  # 'coc_out' 'harmo_out'
    # kl = kurtolearn(channels=224, window=16, radius=8, w_range=(0, 10))

    process_general(process=stella, src_kw='coc_out', tar_kw='stella_c', dataset_list=dataset_list)



