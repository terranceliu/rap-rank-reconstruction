import numpy as np

x = """
01073005910
02282000100
04013092719
05001480100
06037575801
08005086900
09003492500
10003013604
12103026820
13245001300
15009031103
16087970100
17043840600
18177000700
19153000500
20045000300
21001970200
22071002900
23019000500
24021750100
25025160502
26163504800
27109001502
28059040700
29189220445
30091090200
31109003707
32003003411
33015107100
34033020700
35049000700
36029004200
37081016008
38103960000
39061023001
40131050403
41011000900
42033330800
44005040900
45079011702
46107000100
47065000600
48201220300
49035111302
50019951100
51095080205
53005010701
54099005100
55099970100
56031959100
11001011100
"""

x = np.array(x.split())
print(x)

x = x[:-1]
x = x.reshape(10, 5)
for _x in x:
    _x = ' '.join(_x)
    _x = 'TRACTS+=({})'.format(_x)
    print(_x)

print('\n\n\n')

print('[')
for _x in x:
    _x = ['\'' + y + '\'' for y in _x]
    _x = ', '.join(_x)
    _x += ','
    print(_x)
print(']')

