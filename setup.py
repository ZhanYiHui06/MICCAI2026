from setuptools import setup, find_packages

setup(
    name="sc-unsb",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy",
        "pillow",
        "tqdm",
        "pyyaml",
        "dominate",
        "visdom",
        "scipy",
    ],
    python_requires=">=3.8",
    description="SC-UNSB: Unpaired Neural Schrödinger Bridge for Cell Staining Style Transfer",
    author="SC-UNSB Team",
)
