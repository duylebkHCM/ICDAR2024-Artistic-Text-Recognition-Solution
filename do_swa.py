import torch
import argparse
from pathlib import Path


def do_swa(checkpoint):
    
    skip = []
    
    K = len(checkpoint)
    swa = None
    
    for k in range(K):
        state_dict = torch.load(checkpoint[k], map_location=lambda storage, loc: storage)['state_dict']
        if K==1:
            return state_dict
        if swa is None:
            swa = state_dict
        else:
            for k, v in state_dict.items():
                if any(s in k for s in skip): continue
                swa[k] += v
    
    for k, v in swa.items():
        if any(s in k for s in skip): continue
        try:
            swa[k] /= K
        except:
            swa[k] //= K

    return swa

def main(workdir):
    workdir = Path(workdir)
    # checkpoint_paths = list(workdir.glob("best_ck*.pth"))
    checkpoint_paths = [
        "/data/ocr/duyla4/Research/TEXT_RECOGNITION/WordArt/parseq/outputs/parseq_custom/2024-03-27_12-33-11/checkpoints/epoch=6-step=1302-val_accuracy=92.5752-val_NED=97.5427.ckpt",
        "/data/ocr/duyla4/Research/TEXT_RECOGNITION/WordArt/parseq/outputs/parseq_custom/2024-03-27_12-33-11/checkpoints/epoch=7-step=1488-val_accuracy=92.3077-val_NED=97.4002.ckpt",
        "/data/ocr/duyla4/Research/TEXT_RECOGNITION/WordArt/parseq/outputs/parseq_custom/2024-03-27_12-33-11/checkpoints/epoch=8-step=1674-val_accuracy=92.3077-val_NED=97.4559.ckpt"
        # "experiment_ft_real_s2s_res34_v2/checkpoint/Attention_Seq2Seq/best_ck_loss-0.33856.pth",
        # "experiment_ft_real_s2s_res34_v2/checkpoint/Attention_Seq2Seq/best_ck_loss-0.34019.pth"
    ]

    swa_model = do_swa(checkpoint_paths)

    state_dict = torch.load(checkpoint_paths[0], map_location=lambda storage, loc: storage)

    state_dict['state_dict'] = swa_model

    save_path = workdir.joinpath("swa_epoch678.ckpt")

    torch.save(state_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do SWA')
    parser.add_argument('--workdir', help='work directory',)
    args = parser.parse_args()
    main(args.workdir)
