% `mmt' LaTeX class.
% - modifications by Markus Tatzgern 2016/10/18
%   - adapted to needs of MMT (title page etc.)
%   - based on mmt style of IEEE 
% $Id: mmt.cls 11 2007-04-03 22:25:53Z jpeltier $
% - modifications by Meghan Haley on 2006/07/20
%   - manuscript received changed to 31 March for journal style
% - modifications by Steven Bergner on 2006/06/28
%   - made review and preprint work for journal style
%   - leveraged \firstsection{..} title above double column text
%   - included diamond line (currently by loading .eps file)
%   - adjusted headlines and footer (special on first page, with copyrighttext)
% - modifications by Steven Bergner on 2006/05/21
%   - included 'electronic' option using method from egpubl.cls (c)by D.Fellner
%   - included double column (full width) abstract/keywords/index terms,
%     which can be toggled by (ex/in)clusion of 'journal' document option
%     note: abstract is now a command not an environment, see template.tex
%   - copyrightspace enabled by default with opt. \nocopyrightspace switch
%   - 'CR categories' now called 'Index Terms'
%   - adjusted {sub|subsub|}section headline spacing
%   - adjusted to mmt naming (from tcvg or acm)
% - modification of the acmsiggraph.cls class
% - modifications on 2004/04/06 by Torsten Moeller
%    * many modifications to conform to mmt standard
%      - new heading style
%      - new caption style
%      - new reference style
%      - copyrightspace reduced to 0.5in
% - borrows *heavily* from Daniel Kartch's 'siggraph' class. 
% - also uses pieces of 'apalike.sty' and 'authordate1-4.cls'
% - version 0.1 2001/06/01 Stephen Spencer (spencer@acm.org)
% - version 0.2 2001/10/15 Stephen Spencer
%   - the "\onlineid" variable works with the "review" mode, placing a
%     banner across the top of each page "Online Submission ID 'onlineid'
% - version 0.3 2002/01/11 Stephen Spencer
%   - disabled the cover page option.
% - version 0.4 2002/01/23 Stephen Spencer
%   - based on suggestions from James O'Brien, the following changes:
%     - correction if '\ifcamera' and '\ifcameraelse' commands
%     - page numbering in both review and preprint modes
%     have been made.

% ------------ identification --------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mmt}[2016/10/25 MultiMediaTechnology]

% ------------ initial code --------------

\newif\ifmmt@camera
\newif\ifmmt@preprint
\newif\ifmmt@review
\newif\ifmmt@doublespaced
\newif\ifmmt@wide                    \mmt@widefalse
\newif\ifmmtpaper                     \mmtpaperfalse
\newif\ifmmtmasterthesis              \mmtmasterthesisfalse
\newif\ifmmtbachelorthesis            \mmtbachelorthesisfalse
\newif\ifmmtlanguagegerman            \mmtlanguagegermanfalse
\newif\iftvcgsize                     \tvcgsizefalse

\newcommand{\mmt@columnmode}{}
\newcommand{\mmt@pointsize}{}

% ------------ declaration of options --------------

% 'thesis' option

\DeclareOption{masterthesis}{%
    \mmt@cameratrue%    
    \renewcommand{\mmt@columnmode}{onecolumn}%
    \mmt@doublespacedfalse%
    \mmtmasterthesistrue
    }
    
\DeclareOption{bachelorthesis}{%
    \mmt@cameratrue%    
    \renewcommand{\mmt@columnmode}{onecolumn}%
    \mmt@doublespacedfalse%
    \mmtbachelorthesistrue
    }

% 'journal' option
\DeclareOption{paper}{%
    \mmtpapertrue %
    %\@twosidetrue  \@mparswitchtrue %
    %\ExecuteOptions{tvcgpapersize}    
}

% language option ngerman
\DeclareOption{ngerman}{%    
    \mmtlanguagegermantrue
    \PassOptionsToPackage{main=ngerman, english}{babel}
}
    
\DeclareOption{english}{%    
    \mmtlanguagegermanfalse
    \PassOptionsToPackage{main=english, ngerman}{babel}
}
    
% 'cameraready' option.

\DeclareOption{cameraready}{%
    \mmt@cameratrue%
    \mmt@preprintfalse%
    \mmt@reviewfalse%
    \renewcommand{\mmt@columnmode}{twocolumn}%
    \mmt@doublespacedfalse%
    \renewcommand{\mmt@pointsize}{9pt}}

% 'review' option.

\DeclareOption{review}{%
    \mmt@camerafalse%
    \mmt@preprintfalse%
    \mmt@reviewtrue%
    \renewcommand{\mmt@columnmode}{twocolumn}%
    \mmt@doublespacedfalse%
    \renewcommand{\mmt@pointsize}{9pt}
    }

% 'widereview' option.

\DeclareOption{widereview}{%
    \mmt@camerafalse%
    \mmt@preprintfalse%
    \mmt@reviewtrue%
    \renewcommand{\mmt@columnmode}{onecolumn}%
    \mmt@widetrue%
    \mmt@doublespacedtrue%
    \renewcommand{\mmt@pointsize}{12pt}}

% 'preprint' option.

\DeclareOption{preprint}{%
    \mmt@camerafalse%
    \mmt@preprinttrue%
    \mmt@reviewfalse%
    \renewcommand{\mmt@columnmode}{twocolumn}%
    \mmt@doublespacedfalse%
    \renewcommand{\mmt@pointsize}{9pt}}


%% the following code has partly been adapted from egpubl.cls
%\newif\ifpdf  % determine if we are running PDFLaTeX or not
%\ifx\pdfoutput\undefined \pdffalse % we are not running PDFLaTeX 
%\else 
%  %\pdfoutput=1 % we are running PDFLaTeX 
%  \pdftrue 
%\fi 

\newcommand{\mmtinsertpkg}{}
\DeclareOption{electronic}{%
 \ifpdf 
  \renewcommand{\mmtinsertpkg}{%
   \usepackage[pdftex,
    pagebackref=true,
    colorlinks,linkcolor=blue,citecolor=blue,urlcolor=blue,
    bookmarks=false,
    pdfpagemode=UseNone,
    pdftitle={\@title},
    %pdfauthor={\@shortauthor},
    %pdfsubject={\@title},
    pdfkeywords={\mmt@keywords}]{hyperref}
  }
 \else 
  \renewcommand{\mmtinsertpkg}{%
   \usepackage[dvips,
    pagebackref=true,
    colorlinks,linkcolor=blue,citecolor=blue,urlcolor=blue,
    bookmarks=false,
    pdfpagemode=UseNone,
    %pdftitle={\@title},
    %pdfauthor={authorinfo},
    %pdfsubject={\@title},
    %pdfkeywords={\mmt@keywords}
   ]{hyperref}
  }
 \fi
}
%% end of code adapted from egpubl.cls

% Assume, for the moment, that we're preparing a 'cameraready' version.

\ExecuteOptions{cameraready}

% Here's a warning command for use just below.

\newcommand{\mmt@optwarning}[1]{%
    \ifmmt@camera
        \ClassWarningNoLine{mmt}%
            {#1 option used in camera-ready mode.\MessageBreak
             This violates submission specifications}
    \fi
}

% The 'onecolumn' option doesn't work in 'cameraready' mode.
% --> warning mmt@optwarning

\DeclareOption{onecolumn}{%
    \mmt@optwarning{onecolumn}
    \renewcommand{\mmt@columnmode}{onecolumn}    
}

% The 'twocolumn' option works in 'cameraready' mode.

\DeclareOption{twocolumn}{%
    \renewcommand{\mmt@columnmode}{twocolumn}
    }

% Only the '9pt' size works in 'cameraready' mode.

\DeclareOption{9pt}{%
    \renewcommand{\mmt@pointsize}{9pt}}
\DeclareOption{10pt}{%
    \mmt@optwarning{10pt}
    \renewcommand{\mmt@pointsize}{10pt}}
\DeclareOption{11pt}{%
    \mmt@optwarning{11pt}
    \renewcommand{\mmt@pointsize}{11pt}}
\DeclareOption{12pt}{%
    \mmt@optwarning{12pt}
    \renewcommand{\mmt@pointsize}{12pt}}

% The 'singlespace' option works in 'cameraready' mode.

\DeclareOption{singlespace}{%
    \mmt@doublespacedfalse}

% The 'doublespace' option does not work in 'cameraready' mode.

\DeclareOption{doublespace}{%
    \mmt@optwarning{doublespace}
    \mmt@doublespacedtrue}

% No 'titlepage' option in 'cameraready' mode.

\DeclareOption{titlepage}{%
    \OptionNotUsed%
    \ClassWarningNoLine{mmt}{titlepage option not allowed}}

% No 'landscape' mode in 'cameraready' mode, either.

\DeclareOption{landscape}{%
    \OptionNotUsed%
    \ClassWarningNoLine{mmt}{landscape option not allowed}}

% Pass everything else to the 'article' class, upon which this is based.

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\PassOptionsToClass{\mmt@columnmode}{article}
\PassOptionsToClass{a4paper}{article}
%\ifmmtpaper
\PassOptionsToClass{oneside}{article}
%\else
%\PassOptionsToClass{twoside}{article}
%\fi
\ifdim\mmt@pointsize>9pt
    \PassOptionsToClass{\mmt@pointsize}{article}
\fi
\ifmmtpaper\else
    \PassOptionsToClass{12pt}{article}
\fi

% ------------ package loading --------------

\RequirePackage{babel}
\ifmmtlanguagegerman
    \newcommand{\selectthesislanguage}{\selectlanguage{ngerman}}  
\else
    \newcommand{\selectthesislanguage}{\selectlanguage{english}}  
\fi

\LoadClass{article}

% ------------ main code --------------
\newcommand{\firstsectiontxt}{}
\newcommand{\firstsection}[1]{\renewcommand{\firstsectiontxt}{#1}}

\newcommand{\ifcamera}[1]{\ifmmt@camera #1 \fi}
\newcommand{\ifreview}[1]{\ifmmt@review #1 \fi}
\newcommand{\ifcameraelse}[2]{\ifmmt@camera #1 \else #2 \fi}
\newcommand{\ifreviewelse}[2]{\ifmmt@review #1 \else #2 \fi}

% conference template margins
\ifmmtpaper
\renewcommand{\figurename}{Fig.}

%%\setlength{\parindent}{1em}
\else
% otherwise use the default values of article and new margins
\usepackage[includeheadfoot,margin=2.5cm]{geometry}
\fi



% no page numbers - they are added in production
% \pagestyle{plain}

\newcommand{\mmt@abstxt}{}
\let\origabstract\abstract
\let\endorigabstract\endabstract
\renewcommand{\abstract}[1]{\renewcommand{\mmt@abstxt}{#1}}


% \newenvironment{localsize}[1]
% {%
%   \clearpage
%   \let\orignewcommand\newcommand
%   \let\newcommand\renewcommand
%   \makeatletter
%   \input{size#1.clo}%
%   \makeatother
%   \let\newcommand\orignewcommand
% }
% {%
%   \clearpage
% }


%%% huge block of formatting follows that is only relevant for paper formatting.
\ifmmtpaper % formatting only for paper format

\newcommand{\mmt@setninepoint}{
    \renewcommand\normalsize{%
        \@setfontsize\normalsize\@ixpt\@xpt
        \abovedisplayskip 9\p@ \@plus2\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus3\p@
        \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
        \belowdisplayskip \abovedisplayskip
        \let\@listi\@listI}
    \renewcommand\small{%
        \@setfontsize\small\@viipt\@ixpt
        \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus2\p@
        \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 4\p@ \@plus2\p@ \@minus2\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip}
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize\@viiipt{9.5}%
        \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
        \abovedisplayshortskip \z@ \@plus\p@
        \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
        \def\@listi{\leftmargin\leftmargini
                    \topsep 3\p@ \@plus\p@ \@minus\p@
                    \parsep 2\p@ \@plus\p@ \@minus\p@
                    \itemsep \parsep}%
        \belowdisplayskip \abovedisplayskip}
    \renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
    \renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
    \renewcommand\large{\@setfontsize\large\@xpt\@xiipt}
    \renewcommand\Large{\@setfontsize\Large\@xiipt{14}}
    \renewcommand\LARGE{\@setfontsize\LARGE\@xivpt{18}}
    \renewcommand\huge{\@setfontsize\huge\@xviipt{22}}
    \renewcommand\Huge{\@setfontsize\Huge\@xxpt{25}}
    \selectfont
}
\ifdim\mmt@pointsize=9pt
    \mmt@setninepoint
\fi

\newcommand{\mmt@sectionfont}{}
\newcommand{\sectionfont}[1]{\renewcommand{\mmt@sectionfont}{#1}}
\renewcommand\section{\@startsection{section}{1}{\z@}%
                {-2ex \@plus -1ex \@minus -.2ex}%
                {0.8ex \@plus .2ex}%
                {\reset@font\normalsize\sffamily\bfseries\scshape\mmt@sectionfont}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                {-1.8ex\@plus -1ex \@minus -.2ex}%
                {0.8ex \@plus .2ex}%
                {\reset@font\normalsize\sffamily\bfseries\mmt@sectionfont}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                {-1.8ex\@plus -1ex \@minus -.2ex}%
                {0.8ex \@plus .2ex}%
                {\reset@font\sffamily\normalsize\mmt@sectionfont}}

\renewcommand\paragraph{\@startsection{paragraph}{4}{1em}%
                {1ex \@plus 1ex \@minus.2ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\mmt@sectionfont}}
%\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
%                {3.25ex \@plus1ex \@minus.2ex}%
%                {-1em}%
%                {\reset@font\normalsize\sffamily\bfseries\mmt@sectionfont}}

\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                {3.25ex \@plus1ex \@minus .2ex}%
                {-1em}%
                {\reset@font\normalsize\sffamily\bfseries\mmt@sectionfont}}

\ifmmt@wide\else
%% make captionfont 8pt
  \newcommand{\captionfonts}{\scriptsize\sffamily}
  
  \long\def\@makecaption#1#2{%
    \vskip\abovecaptionskip
    \sbox\@tempboxa{\captionfonts #1\ifmmtpaper. \else: \fi #2}%
    \ifdim \wd\@tempboxa >\hsize
      {\captionfonts #1\ifmmtpaper. \else: \fi #2\par}
    \else
      \global \@minipagefalse
      \def\@figcaptype{figure}
      \hb@xt@\hsize{\ifmmtpaper\ifx\@captype\@figcaptype\else\hfil\fi\else\hfil\fi\box\@tempboxa\hfil}%
    \fi
    \vskip\belowcaptionskip}

%% fix the font size of the bibliography to 8pt
  \newdimen\bibindent
  \newdimen\bibspacing
  \setlength\bibindent{1em}
  \setlength{\bibspacing}{\z@}
  \renewenvironment{thebibliography}[1]
       {\section*{\refname}%
        \scriptsize%
        \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
        \list{\@biblabel{\@arabic\c@enumiv}}%
             {\settowidth\labelwidth{\@biblabel{#1}}%
              \leftmargin\labelwidth
              \advance\leftmargin\labelsep
              \itemsep\bibspacing    % should this be commented out?
              \parsep\bibspacing     % should this be commented out?
              \@openbib@code
              \usecounter{enumiv}%
              \let\p@enumiv\@empty
              \renewcommand\theenumiv{\@arabic\c@enumiv}}%
        \sloppy
        \clubpenalty4000
        \@clubpenalty \clubpenalty
        \widowpenalty4000%
        \sfcode`\.\@m}
       {\def\@noitemerr
         {\@latex@warning{Empty `thebibliography' environment}}%
        \endlist}
\fi

\newcommand{\mmt@empty}{}

\newcommand{\mmt@affiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\mmt@affiliation}{#1}}

\newcommand{\mmt@category}{}
\newcommand{\category}[1]{\renewcommand{\mmt@category}{#1}}
\newcommand{\mmtcategory}[1]{\category{#1}}

%\newcommand{\mmt@format}{}
%\newcommand{\format}[1]{\renewcommand{\mmt@format}{#1}}
%\newcommand{\mmtformat}[1]{\format{#1}}
\newcommand{\mmtformat}[1]
{\@latex@warning{Format specification no longer required.}}

\newcommand{\mmt@teaser}{}
\newcommand{\teaser}[1]{\renewcommand{\mmt@teaser}{#1}}

\newcommand{\mmt@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\mmt@keywords}{#1}}

\newcommand{\mmt@indexterms}{}
\newcommand{\CCScatlist}[1]{\renewcommand{\mmt@indexterms}{#1}}

\newcommand{\CCScat}[4]{%
     #1 [#2]%
     \ifx#3\mmt@empty \else : #3\fi%
     \ifx#4\mmt@empty \else ---#4\fi%
}

\newlength{\titlespace}
\setlength{\titlespace}{0.25in}

\newlength{\teaserspace}
\setlength{\teaserspace}{0.25in}

\newlength{\abstxtspace}
\setlength{\abstxtspace}{0.20in}

\renewcommand{\@maketitle}{%

    \begin{center}%
        {\sffamily\ifmmtpaper\huge\else\LARGE\bfseries\fi%
      \mmt@sectionfont%
      \@title \par}%
        \ifmmtpaper%
          %\vspace{2\baselineskip}%
          \vspace{14pt}%
        \else%
    \vspace{1\baselineskip}\fi%
        \large\sffamily\mmt@sectionfont 
        \begin{tabular}[t]{c}%
            \@author
        \end{tabular}%
        \ifx\mmt@affiliation\mmt@empty \else%
            \par\vspace{1\baselineskip}%
            \mmt@affiliation\par%
        \fi%
       \ifmmtpaper\vspace{0.08in}\else\vspace{\titlespace}\fi%
        \ifx\mmt@teaser\mmt@empty \else%
            \begingroup%
                \def\@captype{figure}%
                \mmt@teaser%
            \endgroup\par%
            \vspace{\teaserspace}%
        \fi%
    \end{center} \par%
    
%     \ifmmtpaper% journal formatting with double column abstract and teaser
%         {\scriptsize\sffamily%\renewcommand{\baselinestretch}{1.1}
%         \leftskip = 0.33in \rightskip = \leftskip%
%         \ifx\mmt@abstxt\mmt@empty \else%
%              \begingroup%
%                  {\bfseries Abstract}---\mmt@abstxt%
%              \endgroup\par%
%          \fi%
%          \ifx\mmt@keywords\mmt@empty \else%
%              \begingroup%
%                  %{\normalsize\mmt@absfont {\bfseries Keywords - } \mmt@keywords}%
%                  \vspace{0.5\baselineskip}%
%                  \par\noindent \textbf{Index Terms}---\mmt@keywords%
%              \endgroup\par%
%          \fi%
% %%          \ifx\mmt@indexterms\mmt@empty \else%
% %%              \begingroup%
% %%      %            {\normalsize\mmt@absfont {\bfseries Index Terms - } %
% %%      %             \mmt@indexterms}%
% %%                  \vspace{0.5\baselineskip}%
% %%                  \par\noindent \textbf{Index Terms -} \mmt@indexterms%
% %%              \endgroup\par%
% %%          \fi%
%          }%
% 	\begin{center}\includegraphics{diamondrule.eps}\end{center}
%         %\vspace{\abstxtspace}%
% 	\ifx\firstsectiontxt\mmt@empty \else
% 	  \section{\firstsectiontxt}
% 	\fi
%      \fi%
}

\let\mmt@origmaketitle\maketitle
\let\mmt@origand\and
\renewcommand{\maketitle}{%
    \let\mmt@title\@title%
    \let\mmt@author\@author%
    \mmt@origmaketitle%
    \thispagestyle{empty}%
    \ifmmt@doublespaced%
        \renewcommand{\baselinestretch}{1.66}\selectfont%
    \fi%
    \ifmmtpaper        
        \ifx\mmt@abstxt\mmt@empty \else%
             \begingroup%
                 \begin{origabstract} \mmt@abstxt \end{origabstract} %
             \endgroup\par%
             %\vspace{\abstxtspace}%
         \fi%
         \ifx\mmt@keywords\mmt@empty \else%
             \begingroup%
                 %{\normalsize\mmt@absfont {\bfseries Keywords - } \mmt@keywords}%
                 \vspace{0.5\baselineskip}%
                 \par\noindent \textbf{Keywords: } \mmt@keywords%
             \endgroup\par%
             %\vspace{\abstxtspace}%
         \fi%
         \ifx\mmt@indexterms\mmt@empty \else%
             \begingroup%
     %            {\normalsize\mmt@absfont {\bfseries Index Terms - } %
     %             \mmt@indexterms}%
                 \vspace{0.5\baselineskip}%
                 \par\noindent \textbf{Index Terms:} \mmt@indexterms%
             \endgroup\par%
             %\vspace{\abstxtspace}%
         \fi%
     \fi%
     \ifx\firstsectiontxt\mmt@empty \else
       \ifmmtpaper
         \section*{}
         \vskip -1.5em
       \else
         \section{\firstsectiontxt}
       \fi
     \fi
}
\else
% define maketitle to be empty
\renewcommand{\maketitle}{}

\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 2pt minus 1pt}

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3} 

\fi % formatting only for paper format

% a check to determine the columnmode
\newif\ifmmttwocolumn
\@ifclasswith{article}{twocolumn}{\mmttwocolumntrue}{\mmttwocolumnfalse}
\newcommand{\mmtcolumnmode}{\ifmmttwocolumn 
    \twocolumn
\else
    \onecolumn
\fi}

% End of file 'mmt_bachelor.cls'.

