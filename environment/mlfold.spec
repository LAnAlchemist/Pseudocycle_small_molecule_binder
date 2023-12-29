Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
	mkdir -p $APPTAINER_ROOTFS/opt/conda/envs/env/
	rsync -a --no-o --no-g /software/conda/envs/mlfold/ $APPTAINER_ROOTFS/opt/conda/envs/env/
	rsync -a --no-o --no-g /usr/local/cuda/lib/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
	rsync -a --no-o --no-g /usr/local/cuda/bin/ $APPTAINER_ROOTFS/usr/bin/

%files
	/etc/localtime

%post
	# Common symlinks
	ln -s /net/databases /databases
	ln -s /net/software /software
	ln -s /home /mnt/home
	ln -s /projects /mnt/projects
	ln -s /net /mnt/net

%environment
	export PATH=$PATH:/usr/local/cuda/bin:/opt/conda/envs/env/bin

%runscript
        /opt/conda/envs/env/bin/python "$@"

%labels
	Env Name mlfold
	Env /software/conda/envs/mlfold

