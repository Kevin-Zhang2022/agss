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
    if wav_path.split('/')[2] == 'pump':
        y = crop_pad(y, 44100*10)
    if wav_path.split('/')[2] == 'engine':
        y = crop_pad(y, 44100*3)


    y = (y-y.mean())/(y.std()+1e-5)
    y = coc(y)
    y = resize(y, output_shape=(224, 224))
    y = an(y)

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

def save_cocgram_analog(wav_path, coc, an):
    y, sr = librosa.load(wav_path, sr=44100)
    if wav_path.split('/')[2] == 'us8k' or 'rus8k':
        y = crop_pad(y, 44100*4)
    if wav_path.split('/')[2] == 'esc10':
        y = crop_pad(y, 44100*5)

    y = (y-y.mean())/(y.std()+1e-5)
    y = coc(y)
    y = resize(y, output_shape=(224, 224))

    wave_path_img = wav_path.replace('audio', 'img/coc_out_analog').replace('.wav', '.jpg')
    wav_dir_img = os.path.dirname(wave_path_img)
    os.makedirs(wav_dir_img, exist_ok=True)

    wave_path_npy = wav_path.replace('audio', 'npy/coc_out_analog').replace('.wav', '.npy')
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


def save_analog_to_spike(analog_path,coc,an):

    y = np.load(analog_path)
    y = an(y)

    wave_path_img = analog_path.replace('npy/coc_out_analog', 'img/coc_out').replace('.npy', '.jpg')
    wav_dir_img = os.path.dirname(wave_path_img)
    os.makedirs(wav_dir_img, exist_ok=True)

    wave_path_npy = analog_path.replace('npy/coc_out_analog', 'npy/coc_out')
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

def analog_to_spike_process(dataset_list=['esc10', 'us8k']):
    for dataset in dataset_list:
        src_dir = '../data/'+dataset+'/npy/coc_out_analog'
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = natsorted(dirs)
            files = natsorted(files)
            for i, file in enumerate(files):
                if file.endswith('.npy'):
                    wav_path = os.path.join(root, file).replace('\\', '/')
                    save_analog_to_spike(wav_path, coc, an)
                    print(wav_path, '', i)
                    # print(f'Saved spectrogram for {wav_path} to {tar_dir}')
    print('end')

def save_general(file_path, process, src_kw, tar_kw):
    y = np.load(file_path)
    y = process(y)

    # file_path_img = file_path.replace(f'npy/{src_kw}', f'img/{tar_kw}').replace('.npy', '.jpg')
    # file_dir_img = os.path.dirname(file_path_img)
    # os.makedirs(file_dir_img, exist_ok=True)

    file_path_npy = file_path.replace(f'npy/{src_kw}', f'npy/{tar_kw}')
    file_dir_npy = os.path.dirname(file_path_npy)
    os.makedirs(file_dir_npy, exist_ok=True)

    # plt.figure(figsize=(6.4, 4.8))
    # plt.imshow(y, aspect='auto')
    # plt.xlabel('Time frame')  # win = (2048/22.05k) ~9.3ms  512 hop 2.3ms
    # plt.ylabel('Frequency Channel')
    # plt.title(file_dir_img.split('/')[-1])
    # plt.savefig(file_path_img)
    # plt.close('all')

    np.save(file_path_npy, y)


def coc_process(dataset_list=['esc10', 'us8k']):
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


def coc_process_analog(dataset_list=['esc10', 'us8k', 'pump']):
    for dataset in dataset_list:
        src_dir = '../data/'+dataset+'/audio'
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = natsorted(dirs)
            files = natsorted(files)
            for i, file in enumerate(files):
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file).replace('\\', '/')
                    save_cocgram_analog(wav_path, coc, an)
                    print(wav_path, '', i)
                    # print(f'Saved spectrogram for {wav_path} to {tar_dir}')
    print('end')


def general_process(process, src_kw='coc_out', tar_kw='harmo_out', datasets=None):

    for dataset in datasets:
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

def coc_stack(x):
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, x.shape[1], axis=0)
    return x

    # plt.imshow(x[1],aspect='auto')
    # plt.plot()

# def rename_files()


if __name__ == "__main__":
    matplotlib.use('Agg')
    # matplotlib.use('TkAgg')
    coc = cochlea(channels=224, window=441)
    an = auditorynerve(uth=0.05)
    # hl = harmolearn(channels=224, window=16, w_range=(-1, 1))
    # f = coc_stack
    coc_process(dataset_list=['engine'])
    # coc_process_analog(dataset_list=['rus8k'])
    # analog_to_spike_process(dataset_list=['rus8k'])
    # a = 10
    # general_process(process=hl, src_kw='coc_out', tar_kw='harmo_out', datasets=['us8k', 'motor', 'car'])
    # general_process(process=f, src_kw='coc_out', tar_kw='coc_stack', datasets=['us8k', 'motor', 'car'])


