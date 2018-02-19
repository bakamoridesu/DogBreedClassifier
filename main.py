# -*- coding: utf-8 -*-

'''Entry point to all things to avoid circular imports.'''
from app import app#, freezer
from views import *
if __name__ == '__main__':
    app.run(debug = True)