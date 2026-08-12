import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Octree Scene Diffusion Unified Launcher")
    
    # 1. Define the stage/model to run
    parser.add_argument("--stage", type=str, required=True, 
                        choices=["str_vae", "str_diff", "sem_vae", "sem_diff"],
                        help="Which part of the pipeline to train")
    
    # 2. Define the config (optional, uses defaults if not provided)
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    # Mapping stages to their specific scripts
    script_map = {
        "str_vae": "train_scripts/train_structure_vae.py",
        "str_diff": "train_scripts/train_structure_diffusion.py",
        "sem_vae": "train_scripts/train_sem_vae.py",
        "sem_diff": "train_scripts/train_sem_diffusion.py"
    }

    # Default config mapping
    config_map = {
        "str_vae": "configs/structure_vae_config.yaml",
        "str_diff": "configs/structure_diffusion_config.yaml",
        "sem_vae": "configs/sem_vae_config.yaml",
        "sem_diff": "configs/sem_diffusion_config.yaml"
    }

    target_script = script_map[args.stage]
    target_config = args.config if args.config else config_map[args.stage]

    print(f"🚀 Launching {args.stage.upper()}...")
    print(f"📜 Script: {target_script}")
    print(f"⚙️  Config: {target_config}\n" + "-"*30)

    # Execute the training script from the repo root so `models`, `utils`, ...
    # resolve regardless of where the launcher was invoked from.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run([sys.executable, target_script, "--config", target_config],
                       check=True, cwd=repo_root, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {args.stage}")
        sys.exit(1)

if __name__ == "__main__":
    main()