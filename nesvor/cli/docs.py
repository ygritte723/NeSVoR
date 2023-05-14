import types
import warnings
import docutils.nodes
import docutils.parsers.rst
import docutils.utils
import sphinx.writers.text
import sphinx.builders.text
import sphinx.util.osutil

# adapted from https://stackoverflow.com/questions/57119361/convert-restructuredtext-to-plain-text-programmatically-in-python

NOT_DOC = True


def doc_mode():
    global NOT_DOC
    NOT_DOC = False


def not_doc():
    return NOT_DOC


def parse_rst(text: str) -> docutils.nodes.document:
    parser = docutils.parsers.rst.Parser()
    components = (docutils.parsers.rst.Parser,)
    settings = docutils.frontend.OptionParser(
        components=components
    ).get_default_values()
    document = docutils.utils.new_document("<rst-doc>", settings=settings)
    parser.parse(text, document)
    return document


class RST:
    def __init__(self, rst) -> None:
        self.rst = rst

    def __str__(self) -> str:
        return rst2txt(self.rst)


def rst(source: str):
    if not NOT_DOC:
        return source
    else:
        return RST(source)


def rst2txt(source: str) -> str:
    document = parse_rst(source)
    app = types.SimpleNamespace(
        srcdir=None,
        confdir=None,
        outdir=None,
        doctreedir="/",
        events=None,
        config=types.SimpleNamespace(
            text_newlines="native",
            text_sectionchars="=",
            text_add_secnumbers=False,
            text_secnumber_suffix=".",
        ),
        tags=set(),
        registry=types.SimpleNamespace(
            create_translator=lambda self, something, new_builder: sphinx.writers.text.TextTranslator(
                document, new_builder
            )
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder = sphinx.builders.text.TextBuilder(app)
    translator = sphinx.writers.text.TextTranslator(document, builder)
    document.walkabout(translator)
    return str(translator.body)
