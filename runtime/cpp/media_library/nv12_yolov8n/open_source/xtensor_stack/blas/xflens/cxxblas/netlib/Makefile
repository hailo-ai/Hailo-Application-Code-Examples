all : blas/ cblas/
	$(MAKE) -C blas
	$(MAKE) -C cblas

blas/ :
	tar xfz blas.tgz

cblas/ :
	tar xfz cblas.tgz

clean : blas/ cblas/
	$(MAKE) -C blas clean
	$(MAKE) -C cblas clean

distclean : clean
	$(RM) -rf blas/
	$(RM) -rf cblas/
