from setuptools import setup, find_packages
reqs = [
    'simplex_sdk',
    'simplex_base_model'
    'jieba'
    'numpy'
    'sklearn'
]

setup(
    name='car_rewrite_model',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    package_data={'': ['*.json']},
    package_dir={'': '.'},
    url='https://git.aipp.io/ludezhengeigen/car_rewrite_model',
    license='MIT',
    author='ludezheng',
    author_email='ludezheng@aidigger.com',
    zip_safe=True,
    description='car comments rewrite',
    install_requires=reqs
)
