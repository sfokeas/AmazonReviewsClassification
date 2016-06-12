import subprocess
import sys

print len(sys.argv)
if len(sys.argv)!= 6:
        print "usage: application <starting C> <low exp> <high exp> <jar> <config file>"
        sys.exit(1)

startC = float(sys.argv[1])
for c in [startC*10**exp for exp in range(int(sys.argv[2])-1, int(sys.argv[3]))]:
        listPassed =["java","-jar",sys.argv[4], sys.argv[5], str(c)]
        subprocess.Popen(listPassed)
