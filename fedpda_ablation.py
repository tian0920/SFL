import subprocess
import os
import argparse

# 消融实验配置
ABLATION_CONFIGS = {
    # "full": {},

    # "no_align": {
    #     "fedpdav2.lambda_align": 0.0
    # },
    # "no_proto": {
    #     "fedpdav2.lambda_proto": 0.0
    # },
    # "no_gen": {
    #     "fedpdav2.gen_interval": 999999
    # },
    "base": {
        "fedpdav2.lambda_align": 0.0,
        "fedpdav2.lambda_proto": 0.0,
        "fedpdav2.gen_interval": 999999
    }
}


LOG_DIR = "ablation_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def build_override_args(config_dict):
    return [f"{key}={value}" for key, value in config_dict.items()]


def run_mode(mode, overrides):
    log_file = os.path.join(LOG_DIR, f"{mode}.log")
    overrides_list = build_override_args(overrides)
    overrides_list.append(f"hydra.run.dir=outputs/fedpdav2/{mode}")
    cmd = ["python", "main.py", "method=fedpdav2"] + overrides_list

    print(f"\n🧪 Running mode: {mode}")
    print(f"📝 Logging to: {log_file}\n")

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc, open(log_file, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which ablation mode to run (default: all)"
    )
    args = parser.parse_args()

    if args.mode == "all":
        for mode_name, overrides in ABLATION_CONFIGS.items():
            run_mode(mode_name, overrides)
    else:
        run_mode(args.mode, ABLATION_CONFIGS[args.mode])


if __name__ == "__main__":
    main()
