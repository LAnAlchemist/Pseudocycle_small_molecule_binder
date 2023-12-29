Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%files
/etc/localtime
/etc/apt/sources.list
/lib/libgfortran.so.3
/software/rosetta/DAlphaBall.gcc /usr/bin

%post
# Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks for IPD
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

# apt
apt-get update
apt-get install -y python3-pip python-is-python3

# pip packages
pip install \
  ipython \
  pandas \
  numpy \
  matplotlib \
  ipykernel \
  seaborn \
  biopython \
  ml-collections \
  torch

# PyRosetta
pip install pyrosetta -f https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.linux.wheel/

# Clean up
apt-get clean
pip cache purge

%runscript
exec python "$@"

%help
PyTorch environment for MPNN binder design.
Author: Nate Bennett
