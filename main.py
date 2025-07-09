from importlib import import_module
from sys import argv

if __name__ == '__main__':
  print("this is boilerplate because can't import correctly")
  _ = import_module(argv[1].replace("/", ".").strip(".py"))
