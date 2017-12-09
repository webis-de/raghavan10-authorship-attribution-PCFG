#!/bin/bash

export STANFORDTOOLSDIR=$HOME/stylometry/stanford-parser
export CLASSPATH=$STANFORDTOOLSDIR/stanford-parser-full-2017-06-09/stanford-parser.jar:$STANFORDTOOLSDIR/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar:$STANFORDTOOLSDIR/stanford-postagger-full-2017-06-09/stanford-postagger.jar
export STANFORD_MODELS=$STANFORDTOOLSDIR/stanford-postagger-full-2017-06-09/models
