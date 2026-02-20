import librosa
import numpy as np
import os
import glob
import pickle

# Dataset Paths
base_path = "C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Datasets"
dataset_dirs = {
    'ravdess': './Datasets/ravdess',
    'savee': './Datasets/savee',
    'crema_d': './Datasets/crema_d'
}


# Numeric Emotion Mappings for Each Dataset
ravdess_emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

savee_emotion_map = {
    'a': 'angry',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happy',
    'n': 'neutral',
    's': 'sad',
    'su': 'surprise'
}

crema_emotion_map = {
    'NEU': 'neutral',
    'HAP': 'happy',
    'SAD': 'sad',
    'ANG': 'angry',
    'FEA': 'fearful',
    'DIS': 'disgust',
    'SUR': 'surprised'
}

# Final Emotion Label Mapping
final_emotion_map = {
    'happy': 0,
    'sad': 1,
    'angry': 2,
    'neutral': 3
}

#  Fixed Length for Audio Samples
fixed_length = 16000  # 1 second at 16kHz

#  Audio Preprocessing Function
def preprocess_audio(file_path, target_sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        audio = librosa.util.normalize(audio)
        noise = np.random.randn(len(audio))
        audio_noisy = audio + 0.005 * noise  # Noise Injection

        # Pad or truncate audio to fixed length
        if len(audio_noisy) < fixed_length:
            audio_noisy = np.pad(audio_noisy, (0, fixed_length - len(audio_noisy)), 'constant')
        else:
            audio_noisy = audio_noisy[:fixed_length]

        return audio_noisy, sr
    except Exception as e:
        print(f" Error processing audio file {file_path}: {e}")
        return None, None

# Extract Emotion from Filename
def extract_emotion_ravdess(filename):
    emotion_code = filename.split('-')[2]
    return ravdess_emotion_map.get(emotion_code)

def extract_emotion_savee(filename):
    return savee_emotion_map.get(filename[0])

def extract_emotion_crema(filename):
    parts = filename.split('_')
    emotion_code = parts[2]
    return crema_emotion_map.get(emotion_code)

# Process Audio Datasets
def process_audio_dataset(dataset_name):
    data = []
    labels = []
    dataset_dir = dataset_dirs[dataset_name]

    # Verify the existence of files in the directory
    wav_files = glob.glob(f"{dataset_dir}/**/*.wav", recursive=True)
    if not wav_files:
        print(f" No .wav files found in directory: {dataset_dir}")
        return data, labels

    for filepath in wav_files:
        try:
            filename = os.path.basename(filepath)

            # Extract emotion based on dataset
            if dataset_name == 'ravdess':
                emotion = extract_emotion_ravdess(filename)
            elif dataset_name == 'savee':
                emotion = extract_emotion_savee(filename)
            elif dataset_name == 'crema_d':
                emotion = extract_emotion_crema(filename)
            else:
                emotion = None

            # If emotion is valid and mapped, process the audio
            if emotion in final_emotion_map:
                audio, sr = preprocess_audio(filepath)
                if audio is not None:
                    data.append(audio)
                    labels.append(final_emotion_map[emotion])
        except Exception as e:
            print(f" Error processing {filepath}: {e}")
    return data, labels

# Process All Datasets
all_data = []
all_labels = []

for dataset_name in ['ravdess', 'savee', 'crema_d']:
    data, labels = process_audio_dataset(dataset_name)
    all_data.extend(data)
    all_labels.extend(labels)

# Save processed data
output_dir = base_path
output_path = os.path.join(output_dir, 'processed_audio_data.pkl')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump({'features': np.array(all_data), 'labels': np.array(all_labels)}, f)

print(f"\n Preprocessing Complete")
print(f"Total audio samples: {len(all_data)}")
print(f"Total labels: {len(all_labels)}")
print(f"Processed data saved to: {output_path}")


#Loaded audio files using Librosa or a similar library.
#Resampled them to 16kHz as thats what Wav2Vec expects.
#Normalised the audio Matched each waveform with its emotion label.
#Stored the result