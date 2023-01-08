import torch.nn as nn
import torch.nn.functional as F
from config import config, cnn_conf
import torch
import torchaudio
import timm
from timm.models.layers import to_2tuple,trunc_normal_

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 54, 64)
        self.fc2 = nn.Linear(64, config['num_classes'])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def loss_ce(outputs, labels):
    return nn.CrossEntropyLoss(label_smoothing=0.025)(outputs, labels)


def loss_bce(logits, targets):
    return nn.BCEWithLogitsLoss()(logits, targets)


class CNNModel(nn.Module):
    def __init__(self, in_dim):
        super(CNNModel, self).__init__()
        filts, kerns, strds, dense = cnn_conf['filters'], cnn_conf['kernels'], cnn_conf['strides'], cnn_conf['dense']

        assert len(filts) == len(kerns) and len(strds) == len(kerns)

        # setup the encoders
        self.enc0 = ConvBnPool(in_dim,   filts[0], kerns[0], strid=strds[0])
        self.enc1 = ConvBnPool(filts[0], filts[1], kerns[1], strid=strds[1])
        self.enc2 = ConvBnPool(filts[1], filts[2], kerns[2], strid=strds[2])
        self.enc3 = ConvBnPool(filts[2], filts[3], kerns[3], strid=strds[3])
        self.enc4 = ConvBnPool(filts[3], filts[4], kerns[4], strid=strds[4])

        # global average pooling
        self.gap = nn.AvgPool2d(kernel_size=(9, 7))
        self.flat = nn.Flatten()

        # dropout
        self.drop = nn.Dropout(p=0.2)

        # dense layers
        self.d1 = nn.Linear(in_features=filts[-1]*4, out_features=dense[0])
        self.d2 = nn.Linear(in_features=dense[0], out_features=dense[1])

        # final layers
        self.final = nn.Linear(in_features=dense[-1], out_features=config['n_classes'])

    def forward(self, x):
        # encode
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # pooling
        x = self.gap(x)
        x = self.flat(x)
        # linear layer
        x = self.d1(x)
        x = self.drop(x)
        x = self.d2(x)
        x = self.drop(x)
        x = self.final(x)

        return x


class ConvBnPool(nn.Module):
    def __init__(self, in_dim, out_dim, kern, strid, pad=None, activation="relu"):
        super(ConvBnPool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kern, padding="same")
        self.bn = nn.BatchNorm2d(num_features=out_dim)
        self.pool = nn.MaxPool2d(kernel_size=kern, stride=strid)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


def get_model(model="basic", in_dim=1):
    if model == "basic":
        return DummyModel()
    if model == "cnn":
        return CNNModel(in_dim)
    if model == "ast":
        return ASTModel()
    else:
        raise NotImplementedError


class MelDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=22_050,
            n_fft=1024,
            f_min=200,
            f_max=10_000,
            hop_length=512,
            n_mels=64,
            normalized=True)
        self.todb = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, x):
        x = self.mel(x)
        x = self.todb(x)
        return x

### AST model ###


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=152, fstride=16, tstride=16, input_fdim=128, input_tdim=1876, imagenet_pretrain=True, audioset_pretrain=False, model_size='small224', verbose=True):
                                                                      # 10,   10,   1876,    128 for 30s+noise melspecs, but overloads memory
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # the positional embedding
        if imagenet_pretrain == True:
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)
       

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim


    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x