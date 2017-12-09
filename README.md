## raghavan10 - Authorship Attribution Using Probabilistic Context-Free Grammars

This is a reimplementation of the approach to authorship attribution originally described in

> S. Raghavan, A. Kovashka, and R. Mooney. [Authorship Attribution Using Probabilistic Context-Free Grammars
](http://dl.acm.org/citation.cfm?id=1858842.1858850). In Proc. of the ACL 2010 Conference Short Papers (pp. 38-42). Association for Computational Linguistics, 2010 [[paper](http://dl.acm.org/citation.cfm?id=1858842.1858850)]

## Usage

To execute the software, install it and make sure all its dependencies are installed as well; then run the software using the following command:

`. init_environment.sh && python3 raghavan10.py -i <path-to-input-data> -o <output-path>`

## Dependencies

Stanford Parser:

- https://gist.github.com/alvations/e1df0ba227e542955a8a#stanford-parser (update the init_environment.sh)
- https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software#stanford-tagger-ner-tokenizer-and-parser

NLTK:

```
sudo -H pip3 install -U nltk
python3
>>> import nltk
>>> nltk.download('punkt')
```

## License

Copyright (c) 2017 Roland Schmid

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
