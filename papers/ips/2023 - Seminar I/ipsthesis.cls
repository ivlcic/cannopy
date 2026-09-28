%--------------------------------------------------------------------------------------------------
% ipsthesis v1.3 (see the end of the file for modification history)
% Version 1.3 released 2016/11/07
% 
% A class for creating PhD and MSc theses at the Jožef Stefan International Postgraduate School 
% (IPS) by Tea Tušar <tea.tusar@ijs.si>
%
% Partially based on previous IPS templates by Petra Kralj Novak, Erik Zupanič, Andreja Eršte, 
% Vedrana Vidulin and Boštjan Kaluža. The author would also like to acknowledge many helpful tips
% from the http://tex.stackexchange.com/ website. Special thanks to Bogdan Filipič.
%
% *** Bibliographies are be handled with biblatex (not bibtex) and require biber as backend! ***
%
% Options:
% - eng/slo   English/Slovene template (default is eng)
% - phd/msc   Template for the PhD/MSc thesis (default is phd)
% - apa/ieee  Bibliography style - either APA 6th edition or IEEE (default is apa)
% - draft     Draft mode (as the draft mode for the book class, not default)
%--------------------------------------------------------------------------------------------------
%
% CLASS IDENTIFICATION
%--------------------------------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ipsthesis}[2015/04/13 v1.2 IPS PhD and MSc thesis class]
%--------------------------------------------------------------------------------------------------
%
% CLASS OPTIONS
%--------------------------------------------------------------------------------------------------
\DeclareOption{draft}{
  \PassOptionsToClass{\CurrentOption}{book}
}
%
% English/Slovene
\newif \ifeng
%
\DeclareOption{eng}{
  \engtrue
}
\DeclareOption{slo}{
  \engfalse
}
%
% PhD/MSc thesis
\newif \ifphd
%
\DeclareOption{phd}{
  \phdtrue
}
\DeclareOption{msc}{
  \phdfalse
}
%
% APA/IEEE bibliography style
\newif \ifapa
%
\DeclareOption{apa}{
  \apatrue
}
\DeclareOption{ieee}{
  \apafalse
}
%
% Fallback
\DeclareOption*{
  \ClassWarning{ipsthesis}{Unknown option '\CurrentOption'}
}
%
% Execute default options
\ExecuteOptions{eng}
\ExecuteOptions{phd}
\ExecuteOptions{apa}
%
\ProcessOptions\relax
%
\LoadClass[a4paper,oneside,11pt]{book}
%--------------------------------------------------------------------------------------------------
%
% LANGUAGES AND ENCODING
%--------------------------------------------------------------------------------------------------
\ifeng
	\RequirePackage[slovene,american]{babel}
\else
	\RequirePackage[american,slovene]{babel}
\fi
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
%--------------------------------------------------------------------------------------------------
%
% PAGE LAYOUT AND HEADERS
%--------------------------------------------------------------------------------------------------
\RequirePackage[a4paper,top=30mm,bottom=25mm,inner=35mm,outer=25mm,headsep=10mm]{geometry}
%
\RequirePackage{emptypage}
%
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhead{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[RO]{\small \thepage}
\fancyhead[LO]{\small \nouppercase \rightmark}
\fancyhead[LE]{\small \thepage}
\fancyhead[RE]{\small \nouppercase \leftmark}
\fancyfoot{}
\fancypagestyle{plain}{
  \fancyhead{}
  \renewcommand{\headrulewidth}{0pt}
	\fancyhead[RO]{\small \thepage}
	\fancyhead[LE]{\small \thepage}
  \fancyfoot{}
}
%--------------------------------------------------------------------------------------------------
%
% TITLE PAGES
%--------------------------------------------------------------------------------------------------
%
% Definitions
\def\titleEnglish#1{\def\@titleenglish{#1}}
\def\titleSlovene#1{\def\@titleslovene{#1}}
%
\newif \ifsupervisortwo \supervisortwofalse
\newif \ifsupervisorthree \supervisorthreefalse
\def\supervisorOne#1#2{\def\@supervisorone{#1}\def\@supervisoroneaffiliation{#2}}
\def\supervisorTwo#1#2{\def\@supervisortwo{#1}\def\@supervisortwoaffiliation{#2}\supervisortwotrue}
\def\supervisorThree#1#2{\def\@supervisorthree{#1}\def\@supervisorthreeaffiliation{#2}\supervisorthreetrue}
%
\def\evaluationBoardChairman#1#2{\def\@chairman{#1}\def\@chairmanaffiliation{#2}}
\def\evaluationBoardMemberOne#1#2{\def\@memberone{#1}\def\@memberoneaffiliation{#2}}
\def\evaluationBoardMemberTwo#1#2{\def\@membertwo{#1}\def\@membertwoaffiliation{#2}}
%
\def\thesismonth#1{\def\@thesismonth{#1}}
\def\thesisyear#1{\def\@thesisyear{#1}}
%
% Translations
%
\RequirePackage[algochapter,algoruled]{algorithm2e}
%
\def\@phdnameeng{Seminar I}
\def\@phdnameslo{}
\def\@mscnameeng{Master Thesis}
\def\@mscnameslo{Magistrsko delo}
%
\ifeng
	\ifphd
		\def\@thesisname{\@phdnameeng}
		\def\@thesisnameother{\@phdnameslo}
	\else
		\def\@thesisname{\@mscnameeng}
		\def\@thesisnameother{\@mscnameslo}
	\fi
	\def\@titleorig{\@titleenglish}
	\def\@titleother{}
  \def\@schoolname{Jožef Stefan International Postgraduate School}
  \def\@placename{Ljubljana, Slovenia}
  \def\@supervisorname{Supervisor}
  \def\@cosupervisorname{Co-Supervisor}
  \def\@boardname{}
  \def\@chairname{}
  \def\@membername{}
\else
	\ifphd
		\def\@thesisname{\@phdnameslo}
		\def\@thesisnameother{\@phdnameeng}
		\def\@boardname{Komisija za oceno doktorske disertacije}
	\else
		\def\@thesisname{\@mscnameslo}
		\def\@thesisnameother{\@mscnameeng}
		\def\@boardname{Komisija za oceno magistrskega dela}
	\fi
	\def\@titleorig{\@titleenglish}
	\def\@titleother{\@titleenglish}
  \def\@schoolname{Mednarodna podiplomska šola Jožefa Stefana}
  \def\@placename{Ljubljana, Slovenija}
  \def\@supervisorname{Mentor}
  \def\@cosupervisorname{Somentor}
  \def\@chairname{predsednik}
  \def\@membername{član}
	%
	\renewcommand{\figurename}{Slika}%
	\renewcommand{\tablename}{Tabela}%
	\renewcommand{\listalgorithmcfname}{Algoritmi}%
	\renewcommand{\algorithmcfname}{Algoritem}%
	\renewcommand{\algorithmautorefname}{algoritem}%
	\renewcommand{\algorithmcflinename}{vrstica}%
	\renewcommand{\algocf@typo}{}%
	\renewcommand{\@algocf@procname}{Procedura}%
	\renewcommand{\@algocf@funcname}{Funkcija}%
	\renewcommand{\procedureautorefname}{procedura}%
	\renewcommand{\functionautorefname}{funkcija}%
	\renewcommand{\algocf@languagechoosen}{english}%
	%
	\SetKwHangingKw{KwHData}{Podatki$\rightarrow$}
	\SetKwInput{KwIn}{Vhod}%
	\SetKwInput{KwOut}{Izhod}%
	\SetKwInput{KwData}{Podatki}%
	\SetKwInput{KwResult}{Rezultat}%
	\SetKw{KwTo}{do}
	\SetKw{KwRet}{vrni}%
	\SetKw{Return}{vrni}%
	\SetKwBlock{Begin}{začetek}{konec}%
	\SetKwRepeat{Repeat}{ponavljaj}{dokler}%
	%
	\SetKwIF{If}{ElseIf}{Else}{če}{potem}{sicer če}{sicer}{konec}%
	\SetKwSwitch{Switch}{Case}{Other}{preklop}{delaj}{če}{sicer}{konec}{konec}%
	\SetKwFor{For}{za}{delaj}{konec}%
	\SetKwFor{ForPar}{za}{vzporedno delaj}{konec}
	\SetKwFor{ForEach}{za vsak}{delaj}{konec}%
	\SetKwFor{ForAll}{za vse}{delaj}{konec}%
	\SetKwFor{While}{dokler}{delaj}{konec}%
\fi
%
% Produce title pages
\renewcommand{\maketitle}{%
%
% 1st page
% Deleted
%
% 2nd page
\thispagestyle{empty}
\normalsize
\noindent \textbf{\@thesisname}

\noindent \textbf{\@schoolname}

\noindent \textbf{\@placename}

\vspace{15mm}

\noindent \textbf{\@supervisorname:} \@supervisorone, \@supervisoroneaffiliation

\ifsupervisortwo \noindent \textbf{\@cosupervisorname:} \@supervisortwo, \@supervisortwoaffiliation \fi

\ifsupervisorthree \noindent \textbf{\@cosupervisorname:} \@supervisorthree, \@supervisorthreeaffiliation \fi

\vspace{15mm}

\newpage
%
% 3rd page
\thispagestyle{empty}

\noindent \includegraphics[width=\textwidth]{figures/ipslogo}

\vspace{15mm}
	
\noindent \rm \LARGE \@author

\vspace{20mm}

\noindent \LARGE {\MakeUppercase{\@titleorig}}

\vspace{15mm}
		
\noindent \Large {\bfseries \@thesisname}

\vspace{20mm}

\noindent \LARGE {\MakeUppercase{\@titleother}}

\vspace{15mm}

\noindent \Large {\bfseries \@thesisnameother}

\vspace{20mm}

\noindent \large {\bfseries \@supervisorname:} \@supervisorone 

\ifsupervisortwo \vspace{5mm} \noindent \large {\bfseries \@cosupervisorname:} \@supervisortwo \fi

\ifsupervisorthree \vspace{5mm} \noindent \large {\bfseries \@cosupervisorname:} \@supervisorthree \fi

\vspace{6mm}
\vfill

\noindent {\large \@placename, \@thesismonth\ \@thesisyear}
\normalsize
}
%--------------------------------------------------------------------------------------------------
%
% FORMATTING OF PARTS, CHAPTERS AND SECTIONS
%--------------------------------------------------------------------------------------------------
% 
% Parts
\makeatletter
\renewcommand\part{%
  \if@openright
    \clearpage
  \else
    \clearpage
  \fi
  \thispagestyle{empty}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \null\vfil
  \secdef\@part\@spart%
}
\newcommand{\chapteroutsidepart}{%
	\addtocontents{toc}{\protect\addvspace{14pt}}
	\def\toclevel@chapter{-1}\def\toclevel@section{0}\def\toclevel@subsection{1}
}
\makeatother
%
% Chapter titles
\providecommand\phantomsection{}
\makeatletter
\renewcommand{\chaptermark}[1]{%
  \if@mainmatter
    \markboth{\@chapapp\space \thechapter{}. #1}{}%
  \else
    \markboth{#1}{}%
  \fi
}
\renewcommand*\@makechapterhead[1]{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
				\vspace*{2\p@}%
        \huge\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\vphantom{Ay}\par\nobreak
    \vskip 40\p@
  }}
\renewcommand*\@makeschapterhead[1]{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries #1\vphantom{Ay}\par\nobreak
    \vskip 40\p@
  }}
\makeatother
%
% Appendices
\RequirePackage[titletoc]{appendix}
%
% Depth of section numbering
\setcounter{tocdepth}{5}
\setcounter{secnumdepth}{5}
%
% Adjust spacing after abbreviations and symbols chapter titles
\newcommand{\chapteradjust}{%
	\protect\addvspace{-13pt}%
	\noindent%
}
%--------------------------------------------------------------------------------------------------
%
% FORMATTING OF ALGORITHMS
%--------------------------------------------------------------------------------------------------
%
%\RequirePackage[algochapter,algoruled]{algorithm2e} already included above
\RequirePackage{tocloft}
\makeatletter
\begingroup
  \let\newcounter\@gobble
  \let\setcounter\@gobbletwo
  \globaldefs\@ne
  \let\c@loadepth\@ne
	\newlistof{algorithmt}{loa}{\listalgorithmcfname}
\endgroup
\let\l@algocf\l@algorithmt
\makeatother
%--------------------------------------------------------------------------------------------------
%
% FORMATTING OF TOC, TOF, TOL, TOA
%--------------------------------------------------------------------------------------------------
%
\RequirePackage{graphicx}
%
\pdfoptionpdfminorversion 6
\setlength\belowcaptionskip{10pt}
%
\setlength{\cftfigindent}{0pt}
\setlength{\cfttabindent}{0pt}
\setlength{\cftalgorithmtindent}{0pt}
\renewcommand{\cftfigpresnum}{\figurename~} 
\renewcommand{\cfttabpresnum}{\tablename~} 
\renewcommand{\cftalgorithmtpresnum}{\algorithmcfname~} 
\renewcommand{\cftfigaftersnum}{:} 
\renewcommand{\cfttabaftersnum}{:} 
\renewcommand{\cftalgorithmtaftersnum}{:} 
%
% Indentation in TOF, TOL and TOA
\newlength{\mylen} % a "scratch" length
\setlength{\cftalgorithmtnumwidth}{\cftfignumwidth} 
% TOF
\settowidth{\mylen}{\cftfigpresnum\cftfigaftersnum} 
\addtolength{\cftfignumwidth}{\mylen} 
% TOL
\settowidth{\mylen}{\cfttabpresnum\cfttabaftersnum} 
\addtolength{\cfttabnumwidth}{\mylen} 
% TOA
\settowidth{\mylen}{\cftalgorithmtpresnum\cftalgorithmtaftersnum} 
\addtolength{\cftalgorithmtnumwidth}{\mylen} 
%
% Produce the TOC
\newcommand{\maketoc}{%
	\clearpage
	\vspace*{12pt}
	\pdfbookmark[0]{\contentsname}{\contentsname}
	\tableofcontents
	\addtocontents{toc}{\vspace{-12pt}}
}
%
% Produce the LOF
\newcommand{\makelof}{%
	\clearpage
	\vspace*{12pt}
	\phantomsection\addcontentsline{toc}{chapter}{\listfigurename}
	\addtocontents{lof}{\vspace{-10pt}}
	\listoffigures
}
%
% Produce the LOT
\newcommand{\makelot}{%
	\clearpage
	\vspace*{12pt}
	\phantomsection\addcontentsline{toc}{chapter}{\listtablename}
	\addtocontents{lot}{\vspace{-10pt}}
	\listoftables
}
%
% Produce the LOA
\newcommand{\makeloa}{%
	\vspace*{12pt}
	\clearpage
	\phantomsection\addcontentsline{toc}{chapter}{\listalgorithmcfname}
	\addtocontents{loa}{\vspace{-10pt}}
	\listofalgorithms
}
%--------------------------------------------------------------------------------------------------
%
% THEOREMS AND DEFINITIONS
%--------------------------------------------------------------------------------------------------
%
\RequirePackage{amsmath,amssymb,amsthm}
%
\makeatletter
\def\th@plain{%
  \thm@notefont{}%
  \itshape
}
\def\th@definition{%
  \thm@notefont{}%
  \normalfont
}
\makeatother
%
\theoremstyle{plain}
\ifeng
	\newtheorem{theorem}{Theorem}[chapter]
\else
	\newtheorem{theorem}{Izrek}[chapter]
\fi
%
\theoremstyle{definition}
\ifeng
	\newtheorem{definition}{Definition}[chapter]
\else
	\newtheorem{definition}{Definicija}[chapter]
\fi
%--------------------------------------------------------------------------------------------------
%
% REFERENCES
%--------------------------------------------------------------------------------------------------
%
\RequirePackage[%
	unicode,%
	bookmarks,%
	bookmarksopen=true,%
	bookmarksnumbered=true,%
	bookmarksopenlevel=0,%
	breaklinks=true,%
	colorlinks=true,%
	linkcolor=black,%
	urlcolor=black,%
	citecolor=black,%
	anchorcolor=black,%
	pdfview=Fit,%
	pdfstartview=Fit,%
	pdfpagelayout=SinglePage%
]{hyperref} 
%
\RequirePackage[style=american]{csquotes}
\ifapa
	\RequirePackage[backend=biber,style=apa,doi=true,url=true]{biblatex}[2014/06/25]
	\DeclareLanguageMapping{american}{american-apa}
	\DeclareLanguageMapping{slovene}{slovene-apa}
\else
	\RequirePackage[backend=biber,style=ieee,doi=true,url=true]{biblatex}[2014/06/25]
	\defbibenvironment{nolabelbib}
  {\list
     {}
     {\setlength{\leftmargin}{\bibhang}%
      \setlength{\itemindent}{-\leftmargin}%
      \setlength{\itemsep}{\bibitemsep}%
      \setlength{\parsep}{\bibparsep}}}
  {\endlist}
  {\item}
	\RequirePackage{silence}
	\WarningFilter{biblatex}{Setting 'defernumbers=true' recommended}
\fi
\DefineBibliographyStrings{american}{%
  bibliography = {References}
}
\newcommand{\printreferences}{%
	\clearpage
	\phantomsection\addcontentsline{toc}{chapter}{\refname}
	\printbibliography
}
%--------------------------------------------------------------------------------------------------
%
% INDEX
%--------------------------------------------------------------------------------------------------
%
\RequirePackage{makeidx}
\makeatletter
\renewenvironment{theindex}
	 {\if@twocolumn
			\@restonecolfalse
		\else
			\@restonecoltrue
		\fi
		\setlength{\columnsep}{20pt}
		\twocolumn[\vspace*{74\p@}%
			{\parindent \z@ \raggedright
				\normalfont
				\interlinepenalty\@M
				\Huge \bfseries \indexname\vphantom{Ay}\par\nobreak
				\vskip 38\p@
			}]%
		\@mkboth{\MakeUppercase\indexname}%
						{\MakeUppercase\indexname}%
		\thispagestyle{plain}\parindent\z@
		\parskip\z@ \@plus .3\p@\relax
		\let\item\@idxitem
}
%
\makeatother
\makeindex
\newcommand{\printmyindex}{%
	\clearpage
	\phantomsection\addcontentsline{toc}{chapter}{\indexname}
	\printindex
}
%
\endinput
%--------------------------------------------------------------------------------------------------
%
% MODIFICATION HISTORY
%--------------------------------------------------------------------------------------------------
% Version 1.3: 
%  - Renamed commands \month and \year to \thesismonth and \thesisyear to avoid conflicts.
% Version 1.2: 
%  - Added a command for adjusting the vertical positioning in the abbreviations and symbols chapters
% Version 1.1: 
%  - Corrected headers on even pages in the front and back matter. 
%  - Corrected font size to normalsize after the title pages.
%  - Modified handling of titles on the first title page to accomodate long titles. 
%  - Adjusted the indentation in LOF, LOT and LOA.
%  - Removed the pdftex option from hyperref package to improve compatibility with some packages.
%  - Updated the required biblatex version, which includes slovene localization.
%--------------------------------------------------------------------------------------------------