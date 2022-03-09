import sys

sys.path.append("./WER/LRW/lipreading")
sys.path.append("./WER/Visual_Speech_Recognition_for_Multiple_Languages")
import pypesq
import pesq
from pystoi.stoi import stoi
import librosa
import torch.nn.functional as F
import torch.nn as nn
import torch
from speechbrain.pretrained import EncoderDecoderASR
import time


# local
from WER.LRW.lipreading.model import Audio_Model as WerLRW
from WER.LRW.config import default_model_options as WerLRWOptions
from WER.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.subroutines import LipreadingPipeline
import measures

# Calculate PESQ metric
def calculate_pypesq(real, fake, sr):
    # PESQ only work with sampling rates 8000 or 16000, so we must resample
    if sr != 8000:
        real = librosa.resample(real, sr, 8000)
        fake = librosa.resample(fake, sr, 8000)
    return pypesq.pesq(real, fake, 8000)


def calculate_pesq_nb(real, fake, sr):
    # PESQ only work with sampling rates 8000 or 16000, so we must resample
    if sr != 8000:
        real = librosa.resample(real, sr, 8000)
        fake = librosa.resample(fake, sr, 8000)
    return pesq.pesq(8000, real, fake, "nb")


def calculate_pesq_wb(real, fake, sr):
    # PESQ only work with sampling rates 8000 or 16000, so we must resample
    if sr != 16_000:
        real = librosa.resample(real, sr, 16_000)
        fake = librosa.resample(fake, sr, 16_000)
    return pesq.pesq(16_000, real, fake, "wb")


# Calculate STOI metric
def calculate_stoi(real, fake, sr):
    return stoi(real, fake, sr)


def calculate_estoi(real, fake, sr):
    return stoi(real, fake, sr, extended=True)


class WER_Model(nn.Module):
    def __init__(self, dataset):
        super(WER_Model, self).__init__()
        self.dataset = dataset
        if "grid" in dataset:
            self.model = LipreadingPipeline(
                "./WER/Visual_Speech_Recognition_for_Multiple_Languages/configs/GRID_A_WER0.1.ini"
            )
        elif "lrs3" in dataset:
            self.model = LipreadingPipeline(
                "./WER/Visual_Speech_Recognition_for_Multiple_Languages/configs/LRS3_A_WER2.3.ini", device="cuda:0"
            )
        elif "timit" in dataset:
            self.model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-transformer-transformerlm-librispeech",
                savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
            )
        else:
            raise ValueError

    def forward(self, audio_real_path, audio_fake_path):
        if "grid" in self.dataset or "lrs3" in self.dataset:
            preds_real = self.model(audio_real_path, "")
            preds_fake = self.model(audio_fake_path, "")
        else:
            preds_real = self.model.transcribe_file(audio_real_path)
            preds_fake = self.model.transcribe_file(audio_fake_path)
        print(preds_real)
        print(preds_fake)
        return measures.WER(preds_real.lower(), preds_fake.lower())

    def transcribe_batch(self, audio_real, audio_fake, lengths):
        if "grid" in self.dataset or "lrs3" in self.dataset:
            lengths = [l / max(lengths) for l in lengths]
            preds_real, _ = self.model(audio_real, lengths)
            preds_fake, _ = self.model(audio_fake, lengths)
            wer = []
            for i in range(len(preds_real)):
                wer += [measures.WER(preds_real[i].lower(), preds_fake[i].lower())]
            return wer
        else:
            raise NotImplementedError


class LRW_WER_Model(nn.Module):
    def __init__(self):
        super(LRW_WER_Model, self).__init__()
        options = WerLRWOptions()
        self.model = WerLRW(
            hidden_dim=options["hidden-dim"],
            num_classes=options["num-classes"],
            relu_type=options["relu-type"],
            tcn_options=options["tcn_options"],
        )
        checkpoint = torch.load("./WER/LRW/model_best.pth.tar")
        self.model.load_state_dict(checkpoint["state_dict"])

    def forward(self, audio_real, audio_fake, audio_lengths):
        batch_size = audio_fake.size(0)
        # real
        audio_real = (audio_real - torch.mean(audio_real, dim=-1, keepdim=True)) / torch.std(
            audio_real, dim=-1, keepdim=True
        )
        audio_real = audio_real[:, :, : audio_real.size(-1) // 640 * 640]
        logits_real = self.model(audio_real, lengths=audio_lengths)
        _, preds_real = torch.max(F.softmax(logits_real, dim=1).data, dim=1)
        # fake
        audio_fake = (audio_fake - torch.mean(audio_fake, dim=-1, keepdim=True)) / torch.std(
            audio_fake, dim=-1, keepdim=True
        )
        audio_fake = audio_fake[:, :, : audio_fake.size(-1) // 640 * 640]
        logits_fake = self.model(audio_fake, lengths=audio_lengths)
        _, preds_fake = torch.max(F.softmax(logits_fake, dim=1).data, dim=1)
        # WER
        correct = sum([preds_fake[i].item() == preds_real[i] for i in range(batch_size)])
        return ((batch_size - correct) / batch_size).cpu().numpy()