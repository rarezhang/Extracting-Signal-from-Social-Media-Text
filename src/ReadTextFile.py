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
        self.max_col = 0

    def _set_max_col(self):
        with open(self.path, mode='r', encoding='utf-8', errors='ignore') as f:
            # set how many columns based on first line
            self.max_col = len(f.readline().split(self.sep))


    def read(self):
        with open(self.path, mode='r', encoding='utf-8', errors='ignore') as f:
            # read file line by line
            for line in f:
                line = line.split(self.sep)
                yield line

    def read_column(self, col):
        """

        :param col: int, index of 'line' list
        :return:
        """
        self._set_max_col()
        assert isinstance(col, int) and (0 <= col < self.max_col), 'check index'
        for line in self.read():
            yield line[col]