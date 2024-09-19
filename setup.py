from setuptools import setup
from setuptools import find_packages

author = 'Rasmus Partzsch'

# Requirements
install_requires = ['pylandau', 'coloredlogs', 'tables', 'tqdm', 'numba_progress', 'matplotlib', 'cmcrameri', 'pyyaml'
                    ]

version = '0.9'

setup(
    name='pytestbeam',
    version=version,
    description='Pytestbeam',
    url='https://github.com/Silab-Bonn/aidatlu',
    license='License AGPL-3.0 license',
    long_description='.',
    author=author,
    maintainer=author,
    install_requires=install_requires,
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    platforms='posix',
)
