import clip
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

# ----------------------------
# Modelo Clasificador de Audio
# ----------------------------
class AudioClassifier (nn.Module):
    # ----------------------------
    # Construccion del modelo
    # ----------------------------
    def __init__(self, output_dim):
        super().__init__()
        conv_layers = []

        # Primer Bloque Convolucional
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Segundo Bloque Convolucional
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Tercer Bloque Convolucional
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Cuarto Bloque Convolucional
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Clasificador lineal
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=output_dim)

        # Combinación de los bloques convolucionales
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Ejecucion de los bloques convolucionales
        x = self.conv(x)

        # Adaptive pool y flatten para el input de la capa lineal
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Capa lineal
        x = self.lin(x)

        # Output lineal
        return x
        

# ----------------------------
# CLIP classifier
# ----------------------------  
class CLIP_adapted(nn.Module):
    def __init__(self, selected_model, device, output_dim):
        super(CLIP_adapted, self).__init__()

        # Cargar modelo CLIP pre-entrenado
        self.clip_model, self.preprocess = clip.load(selected_model, device) 
        # Capa lineal para conseguir la dimensión adecuada
        self.fc1 = nn.Linear(2*output_dim, output_dim, device=device) 

    def forward(self, x1, x2):
        x = self.clip_model(x1, x2)
        x = torch.hstack((x[0], x[1].T)).to(torch.float32)
        x = self.fc1(x)
        return x
        
        
# ----------------------------
# Audio-CLIP Model Conjunto
# ----------------------------
class AudioCLIP(nn.Module):
    def __init__(self, selected_model, device, output_dim):
        super(AudioCLIP, self).__init__()

        self.clip_model, self.preprocess = clip.load(selected_model, device)
        self.audio_classifier = AudioClassifier(output_dim).to(device)
        self.fc1 = nn.Linear(2*output_dim, output_dim, device=device)

    def forward(self, x1, x2, y1):

        # Primero, obtenemos el output del modelo CLIP
        x = self.clip_model(x1, x2)
        x = torch.hstack((x[0], x[1].T)).to(torch.float32)
        x = self.fc1(x)

        # Segundo, obtenemos la clasificacion de audio
        y = self.audio_classifier(y1)

        # Acabamos combinando los modelos y obteniendo output
        combined = torch.cat([x, y], dim=1)

        return self.fc1(combined)