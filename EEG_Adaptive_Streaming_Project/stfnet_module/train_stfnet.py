import argparse
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from preprocess.SemiMultichannel import GetEEGData, GetEEGData_train, LoadEEGData
from stfnet_model import STFNet
from tools import acc_multichannel, cal_SNR_multichannel, rrmse_multichannel


def parse_args():
    parser = argparse.ArgumentParser(description="Train STFNet in local stfnet_module")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--EEG_path", type=str, required=True)
    parser.add_argument("--NOS_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./results/STFNet_local/")
    parser.add_argument("--log_dir", type=str, default="./results/json_file/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument(
        "--keep_fold_models",
        action="store_true",
        help="Keep per-fold best.pth files (default: remove and keep only best_overall.pth)",
    )
    return parser.parse_args()


def train_one_fold(opts, fold):
    fold_seed = int(opts.seed + fold)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)

    (
        eeg_train_data,
        nos_train_data,
        eeg_val_data,
        nos_val_data,
        eeg_test_data,
        nos_test_data,
    ) = LoadEEGData(opts.EEG_path, opts.NOS_path, fold, n_splits=opts.folds)

    train_data = GetEEGData_train(
        eeg_train_data, nos_train_data, opts.batch_size, device=opts.device
    )
    val_data = GetEEGData(eeg_val_data, nos_val_data, opts.batch_size)
    test_data = GetEEGData(eeg_test_data, nos_test_data, opts.batch_size)

    model = STFNet(data_num=500, emb_size=32, depth=opts.depth, chan=19).to(opts.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30], 0.1)

    save_path = os.path.join(
        opts.save_dir,
        "STFNet",
        f"STFNet_{opts.depth}",
        f"STFNet_{opts.depth}_{opts.epochs}_{fold}",
    )
    os.makedirs(save_path, exist_ok=True)

    best_val_mse = 10.0
    best_acc = 0.0
    best_snr = -1e9
    best_rrmse = 10.0

    best_model_path = os.path.join(save_path, "best.pth")
    with open(os.path.join(save_path, "result.txt"), "w", encoding="utf-8") as f:
        for epoch in range(opts.epochs):
            model.train()
            losses = []

            for batch_id in trange(train_data.len()):
                x, y = train_data.get_batch(batch_id)
                p = model(x)
                loss = ((p - y) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=0)
                if not torch.isfinite(loss):
                    print(f"[warn] skip non-finite train loss at batch {batch_id}")
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss)

            train_data.random_shuffle()
            if len(losses) == 0:
                print(
                    "[warn] no valid training batch in this epoch, stop training early"
                )
                break
            train_loss = torch.stack(losses).mean().item()
            scheduler.step()

            model.eval()
            losses = []
            for batch_id in range(val_data.len()):
                x, y = val_data.get_batch(batch_id)
                x = torch.Tensor(x).to(opts.device)
                y = torch.Tensor(y).to(opts.device)
                with torch.no_grad():
                    p = model(x)
                    loss = ((p - y) ** 2).mean(dim=-1).mean(dim=-1).sqrt().detach()
                    losses.append(loss)
            val_mse = torch.cat(losses, dim=0).mean().item()

            model.eval()
            rrmse, acc, snr = [], [], []
            for batch_id in range(test_data.len()):
                x, y = test_data.get_batch(batch_id)
                x = torch.Tensor(x).to(opts.device)
                y = torch.Tensor(y).to(opts.device)
                with torch.no_grad():
                    p = model(x)
                    p, y = p.cpu().numpy(), y.cpu().numpy()
                    for i in range(p.shape[0]):
                        rrmse.append(rrmse_multichannel(p[i], y[i]))
                        acc.append(acc_multichannel(p[i], y[i]))
                        snr.append(cal_SNR_multichannel(p[i], y[i]))

            test_rrmse = float(np.mean(np.array(rrmse)))
            test_acc = float(np.mean(np.array(acc)))
            test_snr = float(np.mean(np.array(snr)))

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_acc = test_acc
                best_snr = test_snr
                best_rrmse = test_rrmse
                print("Save best result")
                f.write("Save best result\n")
                torch.save(model, best_model_path)

            print(f"train_loss:{train_loss}")
            msg = (
                f"epoch: {epoch:3d}, val_mse: {val_mse:.4f}, "
                f"test_rrmse: {test_rrmse:.4f}, acc: {test_acc:.4f}, snr: {test_snr:.4f}"
            )
            print(msg)
            f.write(msg + "\n")

    os.makedirs(opts.log_dir, exist_ok=True)
    with open(
        os.path.join(opts.log_dir, f"STFNet_{opts.depth}_{opts.epochs}.log"),
        "a+",
        encoding="utf-8",
    ) as fp:
        fp.write(
            f"fold:{fold}, test_rrmse: {best_rrmse:.4f}, acc: {best_acc:.4f}, snr: {best_snr:.4f}\n"
        )

    return {
        "fold": fold,
        "best_val_mse": best_val_mse,
        "best_rrmse": best_rrmse,
        "best_acc": best_acc,
        "best_snr": best_snr,
        "best_model_path": best_model_path,
    }


def main():
    opts = parse_args()
    total_folds = max(1, opts.folds)
    os.makedirs(opts.save_dir, exist_ok=True)
    os.makedirs(opts.log_dir, exist_ok=True)

    fold_results = []
    for fold in range(total_folds):
        print(f"fold:{fold}")
        fold_result = train_one_fold(opts, fold)
        fold_results.append(fold_result)

    if len(fold_results) == 0:
        return

    # 以验证集指标作为主选择标准，额外用 rrmse 做稳定 tie-break
    best_overall = min(
        fold_results, key=lambda r: (r["best_val_mse"], r["best_rrmse"], -r["best_snr"])
    )
    top_best_path = os.path.join(opts.save_dir, "best_overall.pth")
    if os.path.exists(best_overall["best_model_path"]):
        shutil.copy2(best_overall["best_model_path"], top_best_path)
        print(
            "[best] fold={fold}, val_mse={val:.4f}, rrmse={rrmse:.4f}, acc={acc:.4f}, snr={snr:.4f}".format(
                fold=best_overall["fold"],
                val=best_overall["best_val_mse"],
                rrmse=best_overall["best_rrmse"],
                acc=best_overall["best_acc"],
                snr=best_overall["best_snr"],
            )
        )
        print(f"[best] saved to: {top_best_path}")

    if not opts.keep_fold_models:
        removed = 0
        for item in fold_results:
            p = item.get("best_model_path")
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    removed += 1
                except OSError:
                    pass
        print(f"[cleanup] removed per-fold model files: {removed}")


if __name__ == "__main__":
    main()
