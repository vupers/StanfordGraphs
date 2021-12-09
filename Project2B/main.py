import graphReader

def main():
    #fname = 'facebook_combined.txt'
    fname = 'Roadnet_CA.txt'
    #fname = 'as-skitter.txt'
    graphReader.obtainData(fname)

if __name__ == '__main__':
    main()