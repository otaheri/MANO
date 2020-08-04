from setuptools import setup

pyrender_reqs = ['pyrender>=0.1.23', 'trimesh>=2.37.6', 'shapely']
matplotlib_reqs = ['matplotlib']
open3d_reqs = ['open3d-python']

setup(
    name='MANO',
    include_package_data=True,
    #packages=find_packages(),
    description='A Pytorch Inplementation of MANO differentiable hand model',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version='0.1',
    url='https://github.com/OmidThr/MANO',
    author='Omid Taheri',
    author_email='omid.taheri@tuebingen.mpg.de',
    maintainer='Omid Taheri',
    maintainer_email='omid.taheri@tuebingen.mpg.de',
    #keywords=['pip','MANO'],
    install_requires=[
          'numpy>=1.16.2',
          'torch>=1.0.1.post2',
          'torchgeometry>=0.1.2'
      ],
      extras_require={
          'pyrender': pyrender_reqs,
          'open3d': open3d_reqs,
          'matplotlib': matplotlib_reqs,
          'all': pyrender_reqs + matplotlib_reqs + open3d_reqs
      },
      packages=['mano']  
      
    )
