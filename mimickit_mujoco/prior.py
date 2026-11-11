"""Train the upstream TinyMDM diffusion prior on official G1 samples."""

import json
import sys

import torch
import yaml

from . import ROOT


def train_prior(args, device):
    from .cli import read_yaml

    sys.path.insert(0, str(ROOT / "tools/diffusion_model"))
    from learning.tinymdm.tinymdm_model import TinyMDMModel
    from motion_prior_dataset import MotionPriorData

    config = read_yaml("tools/diffusion_model/config/tinymdm_g1.yaml")
    if args.motion:
        config["motion_file"] = str(args.motion)
    config["num_iterations"] = args.iterations or (2 if args.smoke else config["num_iterations"])
    if args.smoke:
        config.update(
            batch_size=4, num_samples_stat=16, num_layers=1, num_attention_heads=2, attention_head_dim=16, output_iter=1
        )
    out = args.out_dir or ROOT / "output/smp_prior_g1"
    if (out / "model.pt").exists():
        raise SystemExit("Prior output already contains model.pt; choose a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    env_config = read_yaml(config["env_config"])
    # Store the actual data source alongside the prior, including user overrides.
    env_config["motion_file"] = config["motion_file"]
    env_file = out / "env_config.yaml"
    env_file.write_text(yaml.safe_dump(env_config, sort_keys=False))
    config["env_config"] = str(env_file)
    dataset = MotionPriorData(config, device)
    config["input_dim"] = dataset.get_obs_space().shape[-1]
    config["input_channel"] = config["input_dim"] // env_config["num_disc_obs_steps"]
    (out / "diffusion_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    model = TinyMDMModel(config, device).to(device)
    model.update_normalizer(dataset.fetch_obs_demo(config["num_samples_stat"]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"])
    losses = []
    model.train()
    with (out / "loss.jsonl").open("w") as log:
        for step in range(config["num_iterations"]):
            sample = dataset.fetch_obs_demo(config["batch_size"]).detach()
            sample = model.normalize(sample.reshape(config["batch_size"], -1, config["input_channel"]))
            loss = model(sample.flatten(1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite G1 diffusion loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if model.model_ema:
                model.ema_dmodel.update()
            losses.append(loss.item())
            if step % config["output_iter"] == 0 or step + 1 == config["num_iterations"]:
                record = {"iteration": step + 1, "loss": sum(losses) / len(losses)}
                print(json.dumps(record))
                log.write(json.dumps(record) + "\n")
                log.flush()
                torch.save(model.state_dict(), out / "model.pt")
                losses.clear()
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "device": device,
                "smoke": args.smoke,
                "iterations": config["num_iterations"],
                "seed": args.seed,
                "motion": config["motion_file"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved G1 prior: {out}; use --prior-dir {out} for SMP")
