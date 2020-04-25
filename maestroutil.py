import datetime


def prefix_timestamp(filename: str):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    if filename.startswith('/'):
        return timestamp + filename
    else:
        return timestamp + '_' + filename
