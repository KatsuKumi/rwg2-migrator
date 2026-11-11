import multiprocessing
import sys

from rwg2migrator.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
