"""Create synthetic parser fixtures, not a photographic quality demonstration.

python examples/make_smoke_fixture.py --out-dir /absolute/new/fixture
Then compile the emitted direct_scene_brief.json with compile_scene_brief.py.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def create_fixture(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.is_absolute() or out_dir.exists():
        raise ValueError("Choose an absolute, new fixture directory.")
    template = json.loads(Path(__file__).with_name("direct_scene_brief.template.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True)
    environment = Image.new("RGB", (480, 640), (22, 37, 69))
    draw = ImageDraw.Draw(environment)
    draw.line((0, 250, 480, 250), fill=(90, 98, 108), width=8)
    draw.line((240, 0, 240, 420), fill=(90, 98, 108), width=8)
    draw.polygon([(0, 430), (480, 400), (480, 640), (0, 640)], fill=(120, 78, 49))
    draw.ellipse((85, 460, 235, 515), fill=(220, 216, 205))
    environment.save(out_dir/"environment.png")
    identity = Image.new("RGB", (240, 320), (160, 170, 165))
    draw = ImageDraw.Draw(identity)
    draw.ellipse((60, 35, 180, 195), fill=(178, 135, 108))
    draw.arc((52, 15, 188, 125), 170, 365, fill=(35, 28, 26), width=35)
    draw.rectangle((40, 195, 200, 320), fill=(39, 44, 38))
    identity.save(out_dir/"assigned_identity.png")
    template["inputs"] = {"environment": str((out_dir/"environment.png").resolve()),
                          "identity_reference": str((out_dir/"assigned_identity.png").resolve())}
    target = out_dir/"direct_scene_brief.json"
    target.write_text(json.dumps(template, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        path = create_fixture(args.out_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+"\n")
    print(json.dumps({"brief": str(path), "synthetic_only": True, "quality_evidence": False}))
