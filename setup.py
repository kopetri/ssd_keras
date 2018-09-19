import setuptools

setuptools.setup(
    name='ssd_keras',
    version='0.0.1',
    description='keras ssd implementation',
    url='https://github.com/kopetri/ssd_keras.git',
    author='Sebastian Hartwig',
    author_email='sebastian.hartwig@uni-ulm.de',
    maintainer='Sebastian Hartwig',
    maintainer_email='sebastian.hartwig@uni-ulm.de',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'Pillow', 'six', 'keras', 'scikit-learn']
)
