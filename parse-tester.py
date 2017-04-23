import sys
from parse import Parser

#This is just for me, so I can test right now - will be removed 
def main():
    if len(sys.argv) < 2:
        print("Usage: python parse.py [filename]")
        return
    p = Parser(sys.argv[1])
    data = p.get_split_data(train_percentage=1.0)
    print(data["test"][0])
    print(data["test"][1])
    print(len([x for x in data["train"][1] if x == 0]))
    print(len([x for x in data["train"][1] if x == 1]))
    print(len(data["test"][1]) + len(data["train"][1]))
    
if __name__ == "__main__": main()
 
