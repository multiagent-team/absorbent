.PHONY: all preview clean

LATEXMK := latexmk -lualatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build
SOURCES := slides.tex absorbent-theme.sty
ASSETS := $(wildcard assets/*)

all: slides.pdf

slides.pdf: $(SOURCES) $(ASSETS)
	$(LATEXMK) slides.tex
	cp build/slides.pdf $@


preview: slides.pdf
	mkdir -p build/preview
	pdftoppm -png -scale-to 1440 slides.pdf build/preview/slide

clean:
	latexmk -C -outdir=build slides.tex
	rm -f slides.pdf
