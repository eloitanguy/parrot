import _ctypes
import json
import re


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


class OneDictPerLine(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        if not isinstance(self.value, list):
            return repr(self.value)
        else:  # Sort the representation of any dicts in the list.
            reps = ('{{{}}}'.format(', '.join(
                ('{!r}: {}'.format(k, v) for k, v in sorted(v.items()))
            )) if isinstance(v, dict)
                    else
                    repr(v) for v in self.value)
            return '[' + ',\n'.join(reps) + ']'


def di(obj_id):
    """ Reverse of id() function. """
    # from https://stackoverflow.com/a/15012814/355230
    return _ctypes.PyObj_FromPtr(obj_id)


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = "@@{}@@"
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, OneDictPerLine)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON repr.

        # Replace any marked-up object ids in the JSON repr with the value
        # returned from the repr() of the corresponding Python object.
        for match in self.regex.finditer(json_repr):
            id = int(match.group(1))
            # Replace marked-up id with actual Python object repr().
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), repr(di(id)))

        return json_repr
