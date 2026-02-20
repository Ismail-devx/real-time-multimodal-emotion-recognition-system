import os
import pandas as pd
import re

# Load and preprocess GoEmotions dataset
def load_and_preprocess_goemotions(file_path, emotion_map):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['text', 'emotion', 'id'])

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['text'] = df['text'].apply(clean_text)

    def map_emotions(emotion):
        for e in emotion.split(','):
            if int(e) in emotion_map:
                return emotion_map[int(e)]
        return None

    df['emotion'] = df['emotion'].apply(map_emotions)
    df = df.dropna(subset=['emotion'])
    return df

# Load and preprocess DailyDialog dataset
def load_and_preprocess_dailydialog(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)

    # Check if the necessary columns are present
    if 'dialog' not in df.columns or 'emotion' not in df.columns:
        print(f"Error: Missing 'dialog' or 'emotion' columns in {file_path}")
        return pd.DataFrame()

    # Extract the 'dialog' column as text and clean it
    df['text'] = df['dialog'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)).lower())

    # Map the emotion from the 'emotion' column
    emotion_map = {
        0: 'neutral',
        1: 'joy',
        2: 'sadness',
        3: 'anger',
        4: 'fear',
        5: 'surprise',
        6: 'disgust'
    }

    # Extract the first emotion from the list (if multiple)
    def map_emotion_list(emotion_list):
        try:
            # Convert string to list of integers
            emotions = list(map(int, re.findall(r'\d+', emotion_list)))
            # Return the first mapped emotion if available
            if emotions:
                return emotion_map[emotions[0]]
            return 'neutral'
        except:
            return 'neutral'

    df['emotion'] = df['emotion'].apply(map_emotion_list)

    # Keep only necessary columns
    return df[['text', 'emotion']]

# Paths to the datasets
goemotions_dir = './Datasets/goemotions/data/'
dailydialog_dir = './Datasets/dailydialog/'

# GoEmotions files
goemotions_files = ['train.tsv', 'dev.tsv', 'test.tsv']

# DailyDialog files
dailydialog_files = ['train.csv', 'validation.csv', 'test.csv']

# Emotion mapping for GoEmotions
emotion_map = {
    0: 'neutral',
    1: 'joy',
    2: 'sadness',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
    6: 'disgust',
    7: 'trust',
    8: 'anticipation'
}

# Combine data from GoEmotions
goemotions_df = pd.DataFrame()
for file in goemotions_files:
    path = os.path.join(goemotions_dir, file)
    if os.path.exists(path):
        goemotions_df = pd.concat([goemotions_df, load_and_preprocess_goemotions(path, emotion_map)], ignore_index=True)
    else:
        print(f"Error: {file} is missing from GoEmotions dataset.")

# Display the total samples for GoEmotions
if not goemotions_df.empty:
    print(f"GoEmotions preprocessing complete. Total samples: {len(goemotions_df)}")

# Combine data from DailyDialog
dailydialog_df = pd.DataFrame()
for file in dailydialog_files:
    path = os.path.join(dailydialog_dir, file)
    if os.path.exists(path):
        dailydialog_df = pd.concat([dailydialog_df, load_and_preprocess_dailydialog(path)], ignore_index=True)
    else:
        print(f"Error: {file} is missing from DailyDialog dataset.")

# Display the total samples for DailyDialog
if not dailydialog_df.empty:
    print(f"DailyDialog preprocessing complete. Total samples: {len(dailydialog_df)}")

# Merge both datasets
if not goemotions_df.empty and not dailydialog_df.empty:
    combined_df = pd.concat([goemotions_df, dailydialog_df], ignore_index=True)
    output_path = './Datasets/processed_text_data.pkl'
    combined_df.to_pickle(output_path)
    print(f"Combined Text Data. Total samples: {len(combined_df)}")
    print(f"Processed text data saved to: {output_path}")
else:
    print("Error: One or both datasets are empty or missing.")


#Loaded emotion-labeled text dataset
#Used Hugging Face’s tokenizer for each sentence.
#Converted labels into IDs so the model could understand them.
#Split the dataset into training and validation sets.
#Saved the result in a format ready for Trainer or PyTorch training loop.