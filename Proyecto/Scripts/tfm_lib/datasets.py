
import torchaudio
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, random_split
from tfm_lib.audio_processing import AudioUtil, AudioAugmentation

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path='', num_classes=40):
        self.data_path = str(data_path)
        self.duration = 4
        self.sr = 16000
        self.channel = 1
        self.shift_pct = 0.7
        self.num_classes = num_classes
        
        # Filtrar las clases seleccionadas
        self.df = self.filter_classes(df, self.num_classes)
        self.classes = list(self.df['classID'].unique())
            
        # Reindexar el DataFrame filtrado para asegurarse de que los índices sean continuos
        self.df.reset_index(drop=True, inplace=True)
    
    # -----------------------------------
    # Numero de instancias en el dataset
    # -----------------------------------
    def __len__(self):
        return len(self.df)
    
    # -----------------------------------
    # Codificacion de las etiquetas
    # -----------------------------------
    def labelencoder(self):
        le = preprocessing.LabelEncoder()
        return le.fit(self.df["classID"])
    
    # -----------------------------------
    # Elemento i-esimo del dataset
    # -----------------------------------
    def __getitem__(self, idx):
          
        audio_file = self.data_path + self.df.loc[idx, 'audio_path']
    
        le = self.labelencoder()
        db_df = self.df.copy()
        db_df['classID'] = le.transform(db_df['classID'])
    
        class_id = db_df.loc[idx, 'classID']
    
        aud = AudioUtil.open(audio_file)
        dur_aud = AudioAugmentation.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    
        return sgram, class_id

    # --------------------------------------------
    # Filtrar solo la cantidad de clases deseadas
    # -------------------------------------------- 
    def filter_classes(self, df, num_classes):
        # Obtener todas las clases únicas
        unique_classes = df['classID'].unique()
        
        # Seleccionar las primeras num_classes clases
        selected_classes = list(unique_classes)[:num_classes]
        
        # Filtrar el DataFrame para incluir solo las clases seleccionadas
        filtered_df = df[df['classID'].isin(selected_classes)]
        
        return filtered_df


class ImageDataset(Dataset):
    def __init__(self, database_df, num_classes=40, image_transform=None):

      # Atributos derivados de los parametros
      self.image_transform = image_transform
      self.num_classes = num_classes
      self.database_info = self.filter_classes(database_df, num_classes)
      self.database_info.reset_index(drop=True, inplace=True)


      # Codificacion de las clases en valores numericos
      le = preprocessing.LabelEncoder()
      self.labelencoder = le.fit(self.database_info["classID"])
      self.classes = self.labelencoder.classes_

    # --------------------------------
    # Numero de elementos del dataset
    # --------------------------------
    def __len__(self):
      return len(self.database_info)

    # --------------------------------
    # Seleccion del elemento i-esimo
    # --------------------------------
    def __getitem__(self, idx):
      # Definición de los paths donde leer imagen y audio del i-esimo dato
      image_file = self.database_info.loc[idx, 'image_path']

      # Obtenemos la etiqueta para el i-esimo dato
      db_df = self.database_info.copy()
      db_df['classID'] = self.labelencoder.transform(db_df['classID'])
      class_id = db_df.loc[idx, 'classID']

      # Obtención del tensor correspondiente a la i-esima imagen
      read_image = Image.open(image_file)
      image = self.image_transform(read_image)

      return image, class_id

    # --------------------------------------------
    # Filtrar solo la cantidad de clases deseadas
    # -------------------------------------------- 
    def filter_classes(self, df, num_classes):
      # Obtener todas las clases únicas
      unique_classes = df['classID'].unique()

      # Seleccionar las primeras num_classes clases
      selected_classes = list(unique_classes)[:num_classes]

      # Filtrar el DataFrame para incluir solo las clases seleccionadas
      filtered_df = df[df['classID'].isin(selected_classes)]

      return filtered_df


# ----------------------------
# Custom Audio-Image Dataset
# ----------------------------
class CustomDataset(Dataset):
    def __init__(self, database_df, num_classes=40, image_transform=None, audio_transform=None):

      # Atributos derivados de los parametros
      self.image_transform = image_transform
      self.audio_transform = audio_transform
      self.num_classes = num_classes
      self.database_info = self.filter_classes(database_df, num_classes)
      self.database_info.reset_index(drop=True, inplace=True)

      # Atributos relacionados con audio
      self.duration = 4
      self.sr = 16000
      self.channel = 1
      self.shift_pct = 0.7

      # Codificacion de las clases en valores numericos
      le = preprocessing.LabelEncoder()
      self.labelencoder = le.fit(self.database_info["classID"])
      self.classes = self.labelencoder.classes_

    # --------------------------------
    # Numero de elementos del dataset
    # --------------------------------
    def __len__(self):
      return len(self.database_info)

    # --------------------------------
    # Selección del elemento i-esimo
    # --------------------------------
    def __getitem__(self, idx):
      # Definición de los paths donde leer imagen y audio del i-esimo dato
      image_file = self.database_info.loc[idx, 'image_path']
      audio_file = self.database_info.loc[idx, 'audio_path']

      # Obtenemos la etiqueta para el i-esimo dato
      db_df = self.database_info.copy()
      db_df['classID'] = self.labelencoder.transform(db_df['classID'])
      class_id = db_df.loc[idx, 'classID']

      # Obtención del tensor correspondiente a la i-esima imagen
      read_image = Image.open(image_file)
      image = self.image_transform(read_image)

      # Obtención del espectrograma del audio correspondiente al i-esimo dato
      aud = AudioUtil.open(audio_file)
      aud = AudioUtil.resample(aud, self.sr)
      aud = AudioUtil.rechannel(aud, self.channel)
      aud = AudioAugmentation.pad_trunc(aud, self.duration)
      sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)

      return image, sgram, class_id
      
    # --------------------------------------------
    # Filtrar solo la cantidad de clases deseadas
    # -------------------------------------------- 
    def filter_classes(self, df, num_classes):
      # Obtener todas las clases únicas
      unique_classes = df['classID'].unique()
      
      # Seleccionar las primeras num_classes clases
      selected_classes = list(unique_classes)[:num_classes]
      
      # Filtrar el DataFrame para incluir solo las clases seleccionadas
      filtered_df = df[df['classID'].isin(selected_classes)]
      
      return filtered_df