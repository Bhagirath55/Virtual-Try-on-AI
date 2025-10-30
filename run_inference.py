# run_inference.py
import argparse
import os
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path.cwd()

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_inputs(person_src, cloth_src, dest_image_name=None, dest_cloth_name=None):
    # Prepare data/test folders
    data_test = ensure_dir(ROOT / "data" / "test")
    image_dir = ensure_dir(data_test / "image")
    cloth_dir = ensure_dir(data_test / "cloth")

    # decide target filenames
    person_src = Path(person_src)
    cloth_src = Path(cloth_src)
    if dest_image_name is None:
        dest_image_name = person_src.name
    if dest_cloth_name is None:
        dest_cloth_name = cloth_src.name

    person_dst = image_dir / dest_image_name
    cloth_dst = cloth_dir / dest_cloth_name

    shutil.copyfile(person_src, person_dst)
    shutil.copyfile(cloth_src, cloth_dst)
    return dest_image_name, dest_cloth_name

def write_pairs(data_list_path, image_name, cloth_name):
    data_list_path = Path(data_list_path)
    data_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_list_path, "w") as f:
        f.write(f"{image_name} {cloth_name}\n")

def run_test(stage, name, data_list_path, checkpoint, datamode="test", workers=1):
    cmd = [
        sys.executable, "test.py",
        "--stage", stage,
        "--name", name,
        "--datamode", datamode,
        "--data_list", str(data_list_path),
        "--workers", str(workers),
        "--checkpoint", str(checkpoint)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def find_tryon_result(image_name, result_root=ROOT/"result", tom_name="TOM", datamode="test"):
    # typical path: result/TOM/test/try-on/<image_name>
    path = Path(result_root) / tom_name / datamode / "try-on" / image_name
    if path.exists():
        return path
    # sometimes file extensions or directories differ — try jpg/png
    for ext in [".jpg", ".png", ".jpeg"]:
        p = path.with_suffix(ext)
        if p.exists():
            return p
    # fallback: list files in try-on folder and try to match base name
    try_on_dir = Path(result_root) / tom_name / datamode / "try-on"
    if try_on_dir.exists():
        candidates = list(try_on_dir.glob("*"))
        if candidates:
            return candidates[0]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("person", help="path to person image (jpg/png)")
    parser.add_argument("cloth", help="path to cloth image (jpg/png)")
    parser.add_argument("--person_name", help="optional filename to use inside data/test/image")
    parser.add_argument("--cloth_name", help="optional filename to use inside data/test/cloth")
    parser.add_argument("--gmm_ckpt", default="checkpoints/GMM/gmm_final.pth")
    parser.add_argument("--tom_ckpt", default="checkpoints/TOM/tom_final.pth")
    parser.add_argument("--skip_gmm", action="store_true", help="skip running GMM if warp already exists")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep_tmp", action="store_true", help="don't remove data/test files after run")
    args = parser.parse_args()

    # 1) copy inputs
    image_name, cloth_name = copy_inputs(args.person, args.cloth, args.person_name, args.cloth_name)
    data_list = ROOT / "data" / "test" / "test_pairs.txt"
    write_pairs(data_list, image_name, cloth_name)

    # 2) run GMM (warp cloth)
    # GMM will write into result/GMM/test/warp-cloth etc.
    if not args.skip_gmm:
        if not Path(args.gmm_ckpt).exists():
            raise FileNotFoundError(f"GMM checkpoint not found: {args.gmm_ckpt}")
        run_test(stage="GMM", name="GMM", data_list_path=data_list, checkpoint=args.gmm_ckpt, datamode="test", workers=args.workers)
    else:
        print("Skipping GMM (user requested).")

    # 3) run TOM using the warped cloth saved to result/GMM/test
    if not Path(args.tom_ckpt).exists():
        raise FileNotFoundError(f"TOM checkpoint not found: {args.tom_ckpt}")
    run_test(stage="TOM", name="TOM", data_list_path=data_list, checkpoint=args.tom_ckpt, datamode="test", workers=args.workers)

    # 4) locate final try-on image
    result_path = find_tryon_result(image_name)
    if result_path:
        print("Try-on result:", result_path)
    else:
        print("Could not find the try-on result. Check result/TOM/test/try-on/")

    # 5) cleanup (optional)
    if not args.keep_tmp:
        try:
            # Remove the test pair files only (optional safety)
            os.remove(ROOT / "data" / "test" / "test_pairs.txt")
            # keep the data/test image/cloth if you want; remove if required:
            # os.remove(ROOT / "data" / "test" / "image" / image_name)
            # os.remove(ROOT / "data" / "test" / "cloth" / cloth_name)
        except Exception:
            pass

if __name__ == "__main__":
    main()
