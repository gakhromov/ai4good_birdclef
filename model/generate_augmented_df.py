import pandas as pd 
from dataset import combine_labels
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
import librosa
import os
import glob 
import numpy as np 

def main():
    root = '../datasets/birdclef21'
    df = pd.read_csv(f'{root}/train_metadata.csv')

    # create new augmented df
    cols = ['primary_label', 'secondary_labels', 'path']
    df_augmented = pd.DataFrame(columns=cols)

    for idx, item in tqdm(df.iterrows(), total=len(df)):
        path_bird = item['filename'].split('.')[0]
        path = os.path.join(root, 'data',item['primary_label'], path_bird)
        augmented_files = glob.glob(f'{path}*')

        if len(augmented_files) == 0: continue

        # add new row to the augmented for each additional file
        for f in augmented_files:
            data = {
                'path': [f.replace('\\', '/')],
                'file': path_bird,
                'primary_label': [item.primary_label],
                'secondary_labels': [item.secondary_labels],
                'score': item.rating if item.rating != 0 else 2.5,
                #'length': librosa.get_duration(filename=f"{root}/data/{path_bird}.ogg", sr=32_000)*32_000
            }
            new = pd.DataFrame(data=data)
            df_augmented = pd.concat((df_augmented, new), axis=0, ignore_index=True)
    
    mapping = df['primary_label'].unique()
    df = df_augmented.copy()
    classes = mapping
    df['label'] = [combine_labels(df['primary_label'][idx], df['secondary_labels'][idx]) for idx in range(len(df))]
    secondary = [np.sum([np.where(item == classes, 1, 0) for item in row], axis=0) for row in df['label']]
    df['sec_enc'] = secondary
    encoder = LabelEncoder()
    primary = encoder.fit_transform(df['primary_label'])
    df['pri_enc'] = primary
    maps = pd.DataFrame(mapping)
    maps.to_csv(f'{root}/mapping.csv')
    df.to_csv(f'{root}/augmented.csv')



if __name__ == "__main__":
    main()