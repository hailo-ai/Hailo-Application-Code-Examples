all : lapack/
	$(MAKE) -C lapack

lapack/ :
	tar xfz lapack.tgz

clean : lapack/
	$(MAKE) -C lapack clean

distclean : clean
	$(RM) -rf lapack/
