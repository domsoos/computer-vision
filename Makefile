PDFLATEX = crun.texlive pdflatex

all: report presentation

report:
	$(PDFLATEX) manuscript/report.tex
	$(PDFLATEX) manuscript/report.tex

presentation:
	$(PDFLATEX) presentation/presentation.tex
	$(PDFLATEX) presentation/presentation.tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.pdf
