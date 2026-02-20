import os
import sys
from pathlib import Path

# Was having error here so used this to add the project directory to the Python path

try:
    project_root = Path(__file__).resolve().parent  # Script directory
except NameError:
    # Fallback for interactive environments; so I manually set project root here
    project_root = Path(r"C:\Users\ismai\PycharmProjects\EmotionRecognitionSystem")

# Add the project root to sys.path to allow imports from Src
sys.path.append(str(project_root))

# Print debug information for system path and working directory
print(f"Current working directory: {os.getcwd()}")
print("System Path:", sys.path)

# Import the real_time_interface function
try:
    from Src.Emotion_Recognition.real_time import real_time_interface
except ImportError as e:
    print(f"Error: Unable to import real_time_interface. Check your imports and folder structure. ({e})")
    sys.exit(1)

# Entry point for the script
if __name__ == "__main__":
    print("Starting main.py...")

    try:
        # Call the main function from the real_time module
        real_time_interface()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
