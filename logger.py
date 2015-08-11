#!/usr/bin/python

from logline import LogLine

class Logger:
    """ This class logs all the messages. """
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.loglines = []
    def addMessage(self, type, metadata = ""):
        logline = LogLine(self.scheduler.time, type, metadata)
        self.loglines.append(logline)
    def indentLine(self, lineStr):
        return "  " + lineStr
    def indent(self, linesStr):
        lines = linesStr.split("\n")
        return "\n".join(map(self.indentLine, lines))
    def getLoglines(self):
        return "[\n" + (",\n".join(map(self.indent, map(str, self.loglines))) + "\n]\n")
