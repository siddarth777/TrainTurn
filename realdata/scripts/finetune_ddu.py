"""Fine-tune the synthetic-trained junction checkpoint on real DDU days (CUDA only)."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from railway_real.ddu import load_real_config
from railway_real.finetune import finetune


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ddu.yaml"))
    p.add_argument("--scenarios", required=True, help="directory written by build_ddu.py")
    p.add_argument("--passes", type=int, help="override finetune.passes")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = load_real_config(a.config)
    if a.passes is not None:
        cfg["finetune"]["passes"] = a.passes
    print(json.dumps(finetune(cfg, Path(a.scenarios), Path(a.output)), indent=2))


if __name__ == "__main__":
    main()
