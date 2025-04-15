from setuptools import setup, find_packages

setup(
    name='indralux',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pandas', 'scikit-image', 'scipy', 'opencv-python', 'matplotlib', 'seaborn'
    ],
    entry_points={
        'console_scripts': [
            'indralux=cli:main'
        ]
    },
)
