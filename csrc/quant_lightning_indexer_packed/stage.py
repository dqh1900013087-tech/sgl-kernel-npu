"""Stage the generated vendor into the wheel; never install into system CANN."""
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    build, destination = (Path(p).resolve() for p in sys.argv[1:])
    packages = list((build / "package-stage").glob("*.run"))
    if len(packages) != 1:
        raise RuntimeError(f"Expected one generated CANN package, found {packages}")
    stage = build / "wheel-stage"
    stage.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", str(packages[0]), "--quiet", "--install-path=" + str(stage)], check=True)
    vendor = stage / "vendors/sgl_qli_packed"
    if not (vendor / "op_api/lib/libcust_opapi.so").is_file():
        raise RuntimeError("Generated vendor has no opapi library")
    if not list((vendor / "op_impl/ai_core/tbe/kernel").rglob("*.o")):
        raise RuntimeError("Generated vendor has no compiled device kernels")
    # Only remove this build's own generated package subtree.
    target = destination / "vendors/sgl_qli_packed"
    if target.is_symlink():
        raise RuntimeError("Refusing symlinked vendor destination")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(vendor, target, symlinks=False)
    for name in ("LICENSE.ops-transformer", "source-manifest.json"):
        source = Path(__file__).parent / name
        if source.exists():
            shutil.copy2(source, destination / name)
    print("Bundled Packed QLI:", target)


if __name__ == "__main__":
    main()
