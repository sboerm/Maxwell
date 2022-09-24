
CFLAGS = -Wall -Werror -O3 -funroll-loops -ffast-math -fopenmp -march=native
#CFLAGS = -Wall -Werror -g -fopenmp -march=native
LDLIBS = -fopenmp -llapack -lblas -lgfortran -lm

GTKFLAGS = $(shell pkg-config --cflags gtk+-3.0)
GTKLIBS = $(shell pkg-config --libs gtk+-3.0)

MODULES = \
	basic.o \
	settings.o \
	grid2d.o \
	edge2d.o \
	node2d.o \
	linalg.o \
	parameters.o

HEADERS = \
	settings.h

USE_CAIRO=1

FIELD_COMPLEX=1
USE_NETCDF=1

CFLAGS += -DUSE_SIMD

ifdef FIELD_COMPLEX
CFLAGS += -DFIELD_COMPLEX
endif

ifdef USE_CAIRO
CFLAGS += $(shell pkg-config --cflags cairo) -DUSE_CAIRO
LDLIBS += $(shell pkg-config --libs cairo)
endif

ifdef USE_NETCDF
CFLAGS += $(shell pkg-config --cflags netcdf) -DUSE_NETCDF
LDLIBS += $(shell pkg-config --libs netcdf)
MODULES += cdfsupport.o
HEADERS += cdfsupport.h
endif

all: test_edge2d test_node2d test_mgedge2d test_mgnode2d test_jumps2d test_pinvit2d test_grad2d test_lopcg2d test_block2d test_lobpcg2d exp_bloch2d exp_blochsurf eigenview

notes.pdf: notes.tex notes.aux curl_orientation.pdf stencil.pdf arearatio.pdf
	pdflatex $<

notes.aux: notes.tex curl_orientation.pdf stencil.pdf
	pdflatex $<

curl_orientation.pdf: curl_orientation.fig
	fig2dev -Lpdf $< $@

stencil.pdf: stencil.fig
	fig2dev -Lpdf $< $@

arearatio.pdf: arearatio.fig
	fig2dev -Lpdf $< $@

basic.o: basic.c $(HEADERS) basic.h Makefile

settings.o: settings.c $(HEADERS) settings.h Makefile

edge2d.o: edge2d.c $(HEADERS) basic.h grid2d.h edge2d.h Makefile

node2d.o: node2d.c $(HEADERS) basic.h grid2d.h node2d.h edge2d.h Makefile

linalg.o: linalg.c $(HEADERS) linalg.h

parameters.o: parameters.c $(HEADERS) parameters.h

test_edge2d.o: test_edge2d.c $(HEADERS) grid2d.h edge2d.h Makefile

test_edge2d: test_edge2d.o $(MODULES)

test_node2d.o: test_node2d.c $(HEADERS) grid2d.h node2d.h Makefile

test_node2d: test_node2d.o $(MODULES)

test_mgedge2d.o: test_mgedge2d.c $(HEADERS) grid2d.h edge2d.h Makefile

test_mgedge2d: test_mgedge2d.o $(MODULES)

test_mgnode2d.o: test_mgnode2d.c $(HEADERS) grid2d.h edge2d.h Makefile

test_mgnode2d: test_mgnode2d.o $(MODULES)

test_jumps2d.o: test_jumps2d.c $(HEADERS) grid2d.h edge2d.h Makefile

test_jumps2d: test_jumps2d.o $(MODULES)

test_pinvit2d.o: test_pinvit2d.c $(HEADERS) grid2d.h edge2d.h node2d.h basic.h linalg.h Makefile

test_pinvit2d: test_pinvit2d.o $(MODULES)

test_grad2d.o: test_grad2d.c $(HEADERS) grid2d.h edge2d.h node2d.h basic.h linalg.h Makefile

test_grad2d: test_grad2d.o $(MODULES)

test_lopcg2d.o: test_lopcg2d.c $(HEADERS) grid2d.h edge2d.h node2d.h basic.h linalg.h Makefile

test_lopcg2d: test_lopcg2d.o $(MODULES)

test_block2d.o: test_block2d.c $(HEADERS) grid2d.h edge2d.h node2d.h basic.h linalg.h Makefile

test_block2d: test_block2d.o $(MODULES)

test_lobpcg2d.o: test_lobpcg2d.c $(HEADERS) grid2d.h edge2d.h node2d.h basic.h linalg.h Makefile

test_lobpcg2d: test_lobpcg2d.o $(MODULES)

exp_bloch2d.o: exp_bloch2d.c $(HEADERS) grid2d.h edge2d.h node2d.h basic.h linalg.h Makefile

exp_bloch2d: exp_bloch2d.o $(MODULES)

exp_blochsurf.o: exp_blochsurf.c $(HEADERS) grid2d.h edge2d.h node2d.h basic.h linalg.h Makefile

exp_blochsurf: exp_blochsurf.o $(MODULES)

eigenview.o: eigenview.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@ $(GTKFLAGS)

eigenview: eigenview.o Makefile
	$(CC) $(CFLAGS) $< -o $@ $(GTKLIBS) $(LDLIBS)
