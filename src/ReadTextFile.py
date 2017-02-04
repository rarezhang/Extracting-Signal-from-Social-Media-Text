class ReadTextFile:
    """
    read csv-like tweets file
    output file of MongoDB or mysql
    """

    def __init__(self, path, sep=','):
        """
        read file with two columns: label, tweet
        :param path: path to the tweet file
        """
        self.path = path
        self.sep = sep

    def read(self):
        with open(self.path, mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.split(self.sep)
                # label, tweet = line[0], line[1]
                # if self.read_label:
                #     yield label, tweet
                # else:
                #     yield tweet
                yield line

    def read_column(self, col):
        """

        :param col: int, index of 'line' list
        :return:
        """
        assert isinstance(col, int) and 0 <= col, 'check index'
        for line in self.read():
            yield line[col]