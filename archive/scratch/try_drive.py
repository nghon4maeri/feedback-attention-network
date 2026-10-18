import os
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("Drive mounted successfully:", os.path.exists('/content/drive/MyDrive'))
    if os.path.exists('/content/drive/MyDrive/FANet_runs_MFAD/isic-2018_official_a100'):
        print("Drive folder files:", os.listdir('/content/drive/MyDrive/FANet_runs_MFAD/isic-2018_official_a100'))
except Exception as e:
    print("Drive mount exception:", e)
