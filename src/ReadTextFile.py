class ReadTextFile:
    """
    read csv-like tweets file
    output file of MongoDB or mysql
    """

    def __init__(self, path, read_label=True):
        """
        read file with two columns: label, tweet
        :param path: path to the tweet file
        :param readLabel: read Label or not
        """
        self.path = path
        self.read_label = read_label

    def read(self, sep=','):
        with open(self.path, mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.split(sep)
                label, tweet = line[0], line[1]
                if self.read_label:
                    yield label, tweet
                else:
                    yield tweet
