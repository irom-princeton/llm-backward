import setuptools


setuptools.setup(
    name="llm-backward",
    packages=setuptools.find_packages(),
    install_requires=["openai==0.28.0", "numpy", "tqdm", "networkx==3.2.1"],
)
