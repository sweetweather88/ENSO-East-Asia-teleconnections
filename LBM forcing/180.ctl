# &nmhpr khpr=1, hamp=2, xdil=22., ydil=7., xcnt=150., ycnt=-5. &end
# &nmvpr kvpr=2, vamp=-5, vdil=20., vcnt=0.45 &end
DSET ^180.grd
OPTIONS BIG_ENDIAN SEQUENTIAL YREV
TITLE ideal 
UNDEF -999.
XDEF 128 LINEAR 0. 2.81250
YDEF 64  LEVELS -87.864 -85.097 -82.313 -79.526 -76.737 -73.948 -71.158 
-68.368 -65.578 -62.787 -59.997 -57.207 -54.416 -51.626 -48.835 -46.045 
-43.254 -40.464 -37.673 -34.883 -32.092 -29.301 -26.511 -23.720 -20.930 
-18.139 -15.348 -12.558  -9.767  -6.976  -4.186  -1.395   1.395   4.186
  6.976   9.767  12.558  15.348  18.139  20.930  23.720  26.511  29.301
 32.092  34.883  37.673  40.464  43.254  46.045  48.835  51.626  54.416 
 57.207  59.997  62.787  65.578  68.368  71.158  73.948  76.737  79.526 
 82.313  85.097  87.864 
ZDEF 20  LEVELS 0.99500 0.97999 0.94995 0.89988 0.82977 0.74468 
0.64954 0.54946 0.45447 0.36948 0.29450 0.22953 0.17457 0.12440 
0.0846830 0.0598005 0.0449337 0.0349146 0.0248800 0.00829901
TDEF 43 LINEAR 01jan1979 1yr
VARS 4
v      20 99 vor.   forcing [s**-2]
d      20 99 div.   forcing [s**-2]
t      20 99 temp.  forcing [K s**-1]
p       0 99 sfc.Ln(Ps) forcing
ENDVARS