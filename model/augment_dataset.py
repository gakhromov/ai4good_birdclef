import os 
import sys
import subprocess

def main():
    root = '../datasets/birdclef22_augmented/train_audio/'
    folders = os.listdir(root)
    for folder in folders:
        subprocess.run([
            "bsub",
            "-n 2",
            "-R",
            "rusage[mem=8000]",
            "python",
            "test_audioread.py",
            "--folder",
            folder
        ])
        time.sleep(1)


if __name__ == "__main__":
    main()