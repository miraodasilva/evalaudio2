import argparse
import os
import torch
import torchaudio
import torchvision
import numpy as np
from tqdm import tqdm


# local
import eval_metrics

# static variables
sr = 16000

print("Loading args & data & model...")
# load args
parser = argparse.ArgumentParser()
parser.add_argument("--real-folder", "-rf", help="Path to real audio folder")
parser.add_argument("--fake-folder", "-ff", help="Path to fake audio folder")
parser.add_argument("--dataset", "-ds", help="Name of dataset, to pick WER model")

args = parser.parse_args()

# Setting up our device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise Exception("Nao ha GPU")


# validation
wer = []
pypesq = []
pesq_nb = []
pesq_wb = []
stoi = []
estoi = []

if "lrw" in args.dataset:
    wer_model = eval_metrics.LRW_WER_Model()
else:
    wer_model = eval_metrics.WER_Model(args.dataset)
wer_model.to(device)
wer_model.eval()

print("Evaluating... ")
mismatches = 0
for root, _, files in os.walk(args.fake_folder):
    for f in tqdm(files, desc=root.replace(args.fake_folder, "")):
        # only care about wav
        if not f.endswith(".wav"):
            continue
        # loading audio
        audio_fake_path = os.path.join(root, f)
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
        if args.fake_folder == "/vol/paramonos2/projects/rodrigo/feats/v2ajournal_samples/lrwfull" or args.real_folder == "/vol/paramonos/projects/Stavros/lipread_mp4":
            audio_real_path = "/".join(audio_real_path.split("/")[:-1]+ ["test",audio_real_path.split("/")[-1]])
        if os.path.exists(audio_real_path.replace(".wav", ".npz")):
            audio_real_path = audio_real_path.replace(".wav", ".npz")
        elif os.path.exists(audio_real_path.replace(".npz", ".wav")):
            audio_real_path = audio_real_path.replace(".npz", ".wav")
        elif os.path.exists(audio_real_path.replace(".wav", ".mp4")):
            audio_real_path = audio_real_path.replace(".wav", ".mp4")
        else:
            print(audio_real_path)
            mismatches += 1
            print("mismatch!")
            continue
        if audio_real_path.endswith(".npz"):
            audio_real = torch.from_numpy(np.load(audio_real_path)["data"]).view(1, -1)
        elif audio_real_path.endswith(".mp4"):
            _,audio_real,info = torchvision.io.read_video(audio_real_path)
            if info["audio_fps"] != 16_000:
                print("#####Resampling real audio!#####")
                resample = torchaudio.transforms.Resample(audio_real_sr, 16_000)
                audio_real = resample(audio_real)
        else:
            audio_real, audio_real_sr = torchaudio.load(audio_real_path, normalize=True)
            audio_real = audio_real.view(1, -1)
            if audio_real_sr != 16_000:
                print("#####Resampling real audio!#####")
                resample = torchaudio.transforms.Resample(audio_real_sr, 16_000)
                audio_real = resample(audio_real)

        audio_fake = audio_fake.to(device)
        audio_real = audio_real.to(device)

        if audio_real.size(-1) != audio_fake.size(-1):
            diff = abs(audio_real.size(-1) - audio_fake.size(-1))
            # Tolerance of 1000 samples
            print("Cut, real was {} fake was {}!".format(audio_real.size(-1), audio_fake.size(-1)))
            #assert diff < 10000
            min_len = min(audio_real.size(-1), audio_fake.size(-1))
            audio_real = audio_real[:, :min_len]
            audio_fake = audio_fake[:, :min_len]

        # metrics
        if "lrw" in args.dataset:
            wer += [wer_model(audio_real.view(1, 1, -1), audio_fake.view(1, 1, -1), [audio_fake.size(-1)])]
        elif "grid" in args.dataset:
            wer += [wer_model(audio_real_path, audio_fake_path)]
        else:
            wer = -1
        audio_fake_np = audio_fake.detach().squeeze().cpu().numpy()
        audio_real_np = audio_real.detach().squeeze().cpu().numpy()
        try:
            pypesq += [eval_metrics.calculate_pypesq(audio_real_np, audio_fake_np, sr)]
        except:
            print("NaN pypesq!")
            pass
        try:
            pesq_nb += [eval_metrics.calculate_pesq_nb(audio_real_np, audio_fake_np, sr)]
        except:
            print("NaN pesq_nb!")
            pass
        try:
            pesq_wb += [eval_metrics.calculate_pesq_wb(audio_real_np, audio_fake_np, sr)]
        except:
            print("NaN pypesq_nb!")
            pass
        stoi += [eval_metrics.calculate_stoi(audio_real_np, audio_fake_np, sr)]
        estoi += [eval_metrics.calculate_estoi(audio_real_np, audio_fake_np, sr)]

pypesq = np.mean(np.array(pypesq))
pesq_nb = np.mean(np.array(pesq_nb))
pesq_wb = np.mean(np.array(pesq_wb))
stoi = np.mean(np.array(stoi))
estoi = np.mean(np.array(estoi))
wer = np.mean(np.array(wer))

print(
    "PyPESQ: %.4f, PESQ-NB: %.4f, PESQ-WB: %.4f, STOI: %.4f, ESTOI: %.4f, WER: %.4f"
    % (pypesq, pesq_nb, pesq_wb, stoi, estoi, wer)
)
print("Found %s mismatches, ie, samples which were in the fake folder but not in the real folder." % (mismatches))
