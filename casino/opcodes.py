import re
import pathlib

# BEGIN
OP_PASS = 0
OP_PUSH = 1
OP_SUM_START = 51
# END

if __name__ == "__main__":
    ths = pathlib.Path(__file__).absolute()
    pth = (pathlib.Path(__file__).absolute() / '..' / 'core.py').resolve()

    thisFile = open(ths, 'r').readlines()

    with open(pth, 'r') as fin:

        with open('opcodes.py', 'w') as fout:
            for e, lo in enumerate(thisFile):
                fout.write(lo)
                if lo.startswith("# BEGIN"):
                    break

            for l in fin:
                if mtch := re.match("^(_[A-Z_]+)\s?=\s?pyx.declare\(pyx.int,\s?([0-9]+)\)$", l):
                    code = mtch.groups()[0]
                    numb = int(mtch.groups()[1])
                    out = f'OP{code} = {numb}\n'
                    fout.write(out)
                    print(out)
            e += 1
            while e < len(thisFile):
                fout.write(thisFile[e])
                e += 1
