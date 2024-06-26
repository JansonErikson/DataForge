from setuptools import setup, find_packages

setup(
    name='DataForge',
    version='0.1.0',
    description='A package for analyzing and preprocessing DataFrames',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jan Kirschke',
    author_email='jan-kirschke@outlook.de',
    url='https://github.com/JansonErikson/DataForge',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)