#!/bin/bash
gmt begin 
gmt figure figure_14 png E1000
gmt makecpt -Cbukavu -T-200/6000
gmt grdimage @earth_relief_03s -R98/108/20/32 -JM10i -C -Bxa2f0.5 -Bya2f0.5 -BWSne --FONT_ANNOT_PRIMARY=16p,Times-Roman
gmt colorbar -DJBC+w10i/0.2i+h+o0/1c -C -Bxa1000f200+l"Elevation (m)" --FONT_ANNOT_PRIMARY=12p,Times-Roman --FONT_LABEL=14p,Times-Bold
#gmt grdimage CFB_region_1.nc -R98/108/20/32 -JM10i -Ccfb.cpt -nn+a -Q
gmt grdimage CFB_region_2.nc -R98/108/20/32 -JM10i -Ccfb.cpt -nn+a -Q


gmt plot fault.txt -JM10i -W5p,gray0
gmt plot zone.txt -JM10i -W8p,blue,-

gmt plot -Sc0.4c -Gblack << EOF
104.06  30.57     # Chengdu
103.48  29.60     # Emeishan
102.83  24.87     # Kunming
106.63  26.65     # Guiyang
100.27  25.61     # Dali
100.22  26.86     # Lijiang
102.26  27.89     # Xichang
99.15   25.10     # baoshan
105.83  21.03     # Hanoi
107.47  31.21     # Dazhou
EOF

gmt text -F+f18p,Helvetica-Bold,gray1+jCM<< EOF
104.06  30.37      Chengdu
104.28  29.60      Emeishan
102.03  24.87      Kunming
106.63  26.45      Guiyang
100.27  25.81      Dali
101.20  26.86      Lijiang
102.90  27.89      Xichang
99.15   24.70      Baoshan
106.33  21.03      Hanoi
EOF

gmt text -F+f24p,Helvetica-Bold,gray95+jTL -Dj-1.5c/-1.0c << EOF
104.5  28.8  Sichuan Basin
105.5  24.0  Yangtze Block
99.0  22.0   Indochina
98.7   30.0  Tibetan Plateau
100.4  31.5  Songpan-Ganzi Terrane
EOF

echo 98.6 26.0  "Nujiang Fault" | gmt text -F+f18p,Helvetica-Bold,black+jCM+a268
echo 99.4 26.2  "Lacangjiang Fault" | gmt text -F+f18p,Helvetica-Bold,black+jCM+a273
echo 101.8 23.4  "Ailaoshan-Red River Fault" | gmt text -F+f18p,Helvetica-Bold,black+jCM+a315
echo 101.3 28.1  "Lijiang-Xiaojinhe Fault" | gmt text -F+f18p,Helvetica-Bold,black+jCM+a55
echo 102.5 29.0  "Pudu River Fault" | gmt text -F+f18p,Helvetica-Bold,black+jCM+a270
echo 103.5 24.5  "Xiaojiang Fault" | gmt text -F+f18p,Helvetica-Bold,black+jCM+a270

echo 101.6  26.1 | gmt plot -Sa1.5c -W2p,blue -Gblue 
echo 101.9   26.1 0  14 | gmt plot -Sv1.8c+e -W6p,blue -Gblue 

echo 101.6   25.6  "Inner" | gmt text -F+f22p,Helvetica-Bold,blue+jCM -W1p -Glightblue@20 -C25%/25%
echo 103.8  25.6  "Intermediate" | gmt text -F+f22p,Helvetica-Bold,blue+jCM -W1p -Glightblue@20 -C25%/25%
echo 106.0   25.6  "Outer" | gmt text -F+f22p,Helvetica-Bold,blue+jCM -W1p -Glightblue@20 -C25%/25%



gmt inset begin -DjTR+w3i/3i+o0.2c -F+gwhite@70+p2p
gmt coast -Rg -Bg -JG100/30/3i -Gdarkgray -Slightgray -W1/0.5p,black -A5000 -t5
gmt plot -JG100/30/3i -W2p,red << EOF
98 20
108 20
108 30
98 30
98 20
EOF
gmt inset end
gmt basemap -Lx2.6i/0.6i+c103/26+w400k+u+f --FONT_ANNOT_PRIMARY=16p,Times-Roman --MAP_SCALE_HEIGHT=0.5
gmt basemap -Tdg99/31.1+w1.2i+f2+l # North
gmt end
