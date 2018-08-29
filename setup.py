import setuptools
 
setuptools.setup(
    name="k-manolis",
    version="0.0.1",
    author="Manolis Kafouros",
    author_email="kafouros.emm@gmail.com",
    description="Python ml on docker",
    packages=setuptools.find_packages(),
    license='MIT',
      install_requires=['sklearn','numpy','pandas']
)
