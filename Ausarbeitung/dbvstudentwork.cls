%%
%% This is file `dbvstudentwork.cls'
%%
%% IMPORTANT NOTICE:
%% 
%% You are not allowed to change this file.  You may however copy this file
%% to a file with a different name and then change the copy.
%% 
%% You are NOT ALLOWED to distribute this file alone.  You are NOT ALLOWED
%% to take money for the distribution or use of this file (or a changed
%% version) except for a nominal charge for copying etc.
%% 
%% 
\def\filedate{2012/01/02}
\def\fileversion{1.0}
\def\filename{dbvstudentwork.cls}
\NeedsTeXFormat{LaTeX2e}[1997/06/01]

\ProvidesClass{dbvstudentwork}[\filedate\space v\fileversion\space dbvstudentwork class]

\def\BaseClass{scrartcl}

%%%%%%%%%%%%%%% Option Part %%%%%%%%%%%%%%%%%%%%
\RequirePackage{ifthen}

\DeclareOption{10pt}{\def\@fontsize{0}}
\DeclareOption{11pt}{\def\@fontsize{1}}
\DeclareOption{12pt}{\def\@fontsize{2}}

\DeclareOption{english}{\def\@language{english}}
\DeclareOption{german}{\def\@language{german}}
\DeclareOption{ngerman}{\def\@language{ngerman}}
\def\engl{english}

\ifx \@language\engl
\ExecuteOptions{english} \ProcessOptions
\else
\ExecuteOptions{german} \ProcessOptions
\fi

\newif\ifseminar
\DeclareOption{seminar}{\seminartrue}

\newif\ifproject
\DeclareOption{project}{\projecttrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\BaseClass}}
\ProcessOptions\relax

%%%%%%%%%%%%%%% Option Part End %%%%%%%%%%%%%%%%

    \LoadClass[1\@fontsize pt,a4paper,\@language,oneside,bibliography=totoc,listof=totoc,numbers=noenddot]{\BaseClass}[1997/04/16]

    \RequirePackage{setspace}
    \onehalfspacing

    \RequirePackage{graphicx}
    
    \def\dbvstudentworkTitlepage{

	\bgroup
	\def\baselinestretch{1.0}%

	\def\Title##1{\def\title{##1}} \def\title{}
	\def\Subtitle##1{\def\subtitle{##1}} \def\subtitle{}

	\def\FirstName##1{\def\firstName{##1}} \def\firstName{}
	\def\LastName##1{\def\lastName{##1}} \def\lastName{}

	\def\CourseTitle##1{\def\courseTitle{##1}} \def\courseTitle{}
	\def\Advisor##1{\def\advisor{##1}} \def\advisor{}

	\def\Deadline##1{\def\deadline{##1}} \def\deadline{\today}
	\def\SecondInstitute##1{\def\secondInstitute{##1}} \def\secondInstitute{}

    }

    \ifx \@language\engl

	\def\chair{Computer Vision Group}
	\def\department{Department of Mathematics and Computer Science}
	\def\university{Friedrich-Schiller-Universit\"at Jena}

	\def\advisorText{Advisor:}

    \else

	\def\chair{Lehrstuhl f\"ur Digitale Bildverarbeitung}
	\def\department{Fakult\"at f\"ur Mathematik und Informatik}
	\def\university{Friedrich-Schiller-Universit\"at Jena}

	\def\advisorText{Betreuer:}

    \fi

    \def\enddbvstudentworkTitlepage{%

	\enlargethispage{4cm}

	{ % titlepage
	  \thispagestyle{empty}

	  \vfill

	  \begin{center}
	    \ifx \@language\engl \includegraphics[width=50mm]{images/UniJena_BildWortMarke_black.pdf} \else \includegraphics[width=50mm]{images/UniJena_BildWortMarke_black.pdf} \fi
	    \vfill
	    {\Huge \bf \title \\} 
	    \onehalfspacing
	    \ifx \subtitle\empty \else {\Large \bf \subtitle \\} \fi
	    \vfill
	    \ifseminar {\large \bf \ifx \@language\engl Term Paper \else Seminararbeit \fi} \\ \fi
	    \ifproject {\large \bf \ifx \@language\engl Project Work \else Projektarbeit \fi} \\ \fi
	    
	    \vfill
	    \normalsize
	    \ifx \@language\engl submitted by \else vorgelegt von \fi \\
	    {\bf \firstName~\lastName} \\
	    \vfill
	    \ifx \@language\engl written at \else angefertigt am \fi \\
	    {\bf \chair \\ \department \\ \university} \\
	    \ifx \secondInstitute\empty \else \vfill \ifx \@language\engl in cooperation with \else in Zusammenarbeit mit \fi \\ {\bf \secondInstitute} \\ \fi 
	  \end{center}

	  \vfill
	  \noindent
	  \ifseminar {\ifx \@language\engl Seminar:~\courseTitle \else Seminar:~\courseTitle \fi} \\ \fi
	  \ifproject {\ifx \@language\engl Project:~\courseTitle \else Projekt:~\courseTitle \fi} \\ \fi
	  \advisorText~\advisor\\
	  \ifx \@language\engl Date:~\deadline \else Datum:~\deadline \fi
      
	  \vfill
	}

	\clearpage

	\setcounter{page}{1}

	\egroup
    }

    \pagenumbering{arabic}

    \ifx \@language\engl
    \IfFileExists{babel.sty}
    {\RequirePackage[\@language]{babel}[1997/01/23] }
    {\IfFileExists{english.sty}
      {\RequirePackage{english}[1997/05/01]}
      {\ClassError{dbvstudentwork}
	{Neither babel nor english.sty installed !!!}
	{Get babel or english.sty !!!}}}
    \else
    \IfFileExists{babel.sty}
      {\RequirePackage[\@language]{babel}[1997/01/23]}
      {\ClassError{dbvstudentwork}
	{Babel not installed !!!}
	{Get babel package !!!}}
    \fi

\endinput
%%
%% End of file `dbvstudentwork.cls'.
