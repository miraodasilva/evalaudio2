import argparse
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader


# local
import eval_metrics

# static variables
sr = 16000

# Pad a sequence on the right with zeros such that it goes from [len,x,y,z,...] to [desired_len,x,y,z,...]
# To my knowledge there is no pytorch function that does this for any dimensionality
def pad(seq, length):
    return torch.cat([seq, torch.zeros(length - seq.size(0), *seq.size()[1:]).to(seq.device)])


# Pad and stack a sequence of samples into a mini-batch which is representable as a single matrix
def pad_n_stack(seq):
    lengths = [x.size(0) for x in seq]
    # We will pad everything to the maximum length in order to be able to stack the samples
    max_length = max(lengths)

    batch = []
    for sample, length in zip(seq, lengths):
        new_sample = sample
        if length < max_length:
            new_sample = pad(sample, max_length)
        batch += [new_sample]

    return torch.stack(batch), lengths


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, real_folder, fake_folder):
        self.real_folder = real_folder
        self.fake_folder = fake_folder
        self.files = []
        for root, _, files in os.walk(fake_folder):
            for f in files:
                # only care about wav
                if f.endswith(".wav"):
                    self.files += [os.path.join(root, f)]
        self.mismatches = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_fake_path = self.files[idx]
        if audio_fake_path.endswith(".npz"):
            audio_fake = torch.from_numpy(np.load(audio_fake_path)["data"]).view(1, -1)
        else:
            audio_fake, audio_fake_sr = torchaudio.load(audio_fake_path, normalize=True)
            audio_fake = audio_fake.view(1, -1)
            if audio_fake_sr != 16_000:
                print("#####Resampling fake audio!#####")
                resample = torchaudio.transforms.Resample(audio_fake_sr, 16_000)
                audio_fake = resample(audio_fake)
        audio_real_path = audio_fake_path.replace(args.fake_folder, args.real_folder)
        if args.fake_folder == "/vol/paramonos2/projects/rodrigo/feats/v2ajournal_samples/lrwfull":
            audio_real_path = "/".join(audio_real_path.split("/")[:-1] + ["test", audio_real_path.split("/")[-1]])
        if os.path.exists(audio_real_path.replace(".wav", ".npz")):
            audio_real_path = audio_real_path.replace(".wav", ".npz")
        elif os.path.exists(audio_real_path.replace(".npz", ".wav")):
            audio_real_path = audio_real_path.replace(".npz", ".wav")
        else:
            print(audio_real_path)
            self.mismatches += 1
            print("{} mismatches so far!".format(self.mismatches))
            return self.__getitem__(idx + 1)
        if audio_real_path.endswith(".npz"):
            audio_real = torch.from_numpy(np.load(audio_real_path)["data"]).view(1, -1)
        else:
            audio_real, audio_real_sr = torchaudio.load(audio_real_path, normalize=True)
            audio_real = audio_real.view(1, -1)
            if audio_real_sr != 16_000:
                print("#####Resampling real audio!#####")
                resample = torchaudio.transforms.Resample(audio_real_sr, 16_000)
                audio_real = resample(audio_real)

        if audio_real.size(-1) != audio_fake.size(-1):
            diff = abs(audio_real.size(-1) - audio_fake.size(-1))
            # Tolerance of 1000 samples
            print("Cut, real was {} fake was {}!".format(audio_real.size(-1), audio_fake.size(-1)))
            # assert diff < 10000
            min_len = min(audio_real.size(-1), audio_fake.size(-1))
            audio_real = audio_real[:, :min_len]
            audio_fake = audio_fake[:, :min_len]
        return audio_fake.squeeze(), audio_real.squeeze(), audio_fake_path, audio_real_path

    def collate(self, batch):
        audio_fake_paths = []
        audio_real_paths = []
        audios_fake = []
        audios_real = []
        for audio_fake, audio_real, audio_fake_path, audio_real_path in batch:
            audio_fake_paths += [audio_fake_path]
            audio_real_paths += [audio_real_path]
            audios_fake += [audio_fake]
            audios_real += [audio_real]
        audios_fake, audio_lengths = pad_n_stack(audios_fake)
        audios_real, audio_lengths = pad_n_stack(audios_real)
        return {
            "wav_real": audios_real.view(-1, 1, max(audio_lengths)),
            "wav_fake": audios_fake.view(-1, 1, max(audio_lengths)),
            "wav_lengths": audio_lengths,
            "audio_fake_paths": audio_fake_paths,
            "audio_real_paths": audio_real_paths,
        }


print("Loading args & data & model...")
# load args
parser = argparse.ArgumentParser()
parser.add_argument("--real-folder", "-rf", help="Path to real audio folder")
parser.add_argument("--fake-folder", "-ff", help="Path to fake audio folder")
parser.add_argument("--dataset", "-ds", help="Name of dataset, to pick WER model")
parser.add_argument("--batch-size", "-bs", help="Batch size", type=int)

args = parser.parse_args()

# Setting up our device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise Exception("No GPU available.")


# validation
wer = []

if "lrw" in args.dataset:
    wer_model = eval_metrics.LRW_WER_Model()
else:
    wer_model = eval_metrics.WER_Model(args.dataset)
wer_model.to(device)
wer_model.eval()

print("Evaluating... ")
dataset = AudioDataset(args.real_folder, args.fake_folder)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, collate_fn=dataset.collate)

for batch in tqdm(dataloader):
    audio_fake = batch["wav_fake"]
    audio_real = batch["wav_real"]
    lengths = batch["wav_lengths"]
    audio_fake = audio_fake.to(device)
    audio_real = audio_real.to(device)

    # metrics
    if "lrw" in args.dataset:
        wer += [
            wer_model(
                audio_real.view(audio_real.size(0), 1, -1),
                audio_fake.view(audio_fake.size(0), 1, -1),
                [audio_fake.size(-1)],
            )
        ]
    else:
        wer += [wer_model.transcribe_batch(audio_real, audio_fake, lengths)]
wer = np.mean(np.array(wer))

print("WER: %.4f" % (wer))
print(
    "Found %s mismatches, ie, samples which were in the fake folder but not in the real folder." % (dataset.mismatches)
)
