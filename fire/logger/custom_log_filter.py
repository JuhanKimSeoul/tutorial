def filter_maker(logger):
    def filter(record):
        return record.name == logger
    return filter