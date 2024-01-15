#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

