from setuptools import setup, find_packages
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='quantum-mini-apps',
    version='1.0.0',
    author='Pradeep Mantha',
    author_email='pradeepm66@example.com',
    description='This repository contains a framework for developing and benchmarking Quantum Mini-Apps, which are small, self-contained applications designed to evaluate the performance of quantum computing systems and algorithms.',
    packages=find_packages(),
    # install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)