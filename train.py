from process.d_trainfunction import train

datasets_paths = ["data/us8k/npy/coc_out",
                  "data/esc10/npy/coc_out"]

# train(datasets_paths, path2tab='tab', path2trained='bestmodel',
#       trials=2, batch_size=5, num_epochs=2, initial_lr=0.1, step_size=5, test_train=True)

train(datasets_paths, path2tab='tab', path2trained='bestmodel',
      trials=5, batch_size=50, num_epochs=50, initial_lr=0.1, step_size=10)

print('finish')

# import wave
#
# def get_wav_sample_rate_wave(file_path):
#     with wave.open(file_path, 'rb') as wav_file:
#         sample_rate = wav_file.getframerate()
#         return sample_rate
#
# file_path = 'data/esc10/audio/chainsaw/1-19898-A.wav'
# sample_rate = get_wav_sample_rate_wave(file_path)
# print(f'Sample rate: {sample_rate} Hz')