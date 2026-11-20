"""The demo's templates: pages laid out ahead, each a :class:`~pagedemo.document.Document` whose boxes hold a text of
their kind of page (invented, in the Swedish of the archives the model was trained on). A template is a start: every
box of it can be moved, rewritten or removed in the editor."""

from __future__ import annotations

from pagedemo.document import GRID, Box, BoxKind, Document


def _block(
    name: str,
    left: float,
    top: float,
    right: float,
    text: str,
    *,
    lines: int,
    of: int,
    kind: BoxKind = BoxKind.TEXT,
) -> Box:
    """A box ``lines`` line slots tall on a page of ``of`` slots, from ``top``."""
    return Box(id=name, kind=kind, rect=(left, top, right, top + lines * GRID / of), text=text)


def _rule(name: str, left: float, top: float, right: float, bottom: float) -> Box:
    return Box(id=name, kind=BoxKind.RULE, rect=(left, top, right, bottom))


LETTER = Document(
    aspect=0.707,
    lines=30,
    boxes=[
        _block("text_1", 480, 80, 870, "Stockholm den 14 Maji 1789.", lines=1, of=30),
        _block("text_2", 110, 180, 620, "Högädle Herr Häradshöfding,", lines=1, of=30),
        _block(
            "text_3",
            110,
            250,
            870,
            "Sedan jag i går undfick Herr Häradshöfdingens gunstiga skrifwelse af den 2 hujus, har jag ej bordt "
            "underlåta at därpå i ödmjukhet swara, det Länsmannen redan är anbefalld at med första lägenhet infinna "
            "sig wid Tingsstället, och där aflämna de handlingar, hwilka uti målet angående Skattehemmanet Norrgården "
            "blifwit åberopade.\n"
            "Skulle något widare härutinnan fordras, anhåller jag ödmjukast om benägen underrättelse, då jag icke "
            "skall försumma at sådant genast wärkställa.",
            lines=9,
            of=30,
        ),
        _block(
            "text_4",
            440,
            620,
            870,
            "Med wördnad framhärdar\nHögädle Herr Häradshöfdingens\nödmjuke tjenare",
            lines=3,
            of=30,
        ),
        _block("text_5", 530, 770, 870, "Anders Lindqwist", lines=1, of=30),
    ],
)

MINUTES = Document(
    aspect=0.667,
    lines=36,
    boxes=[
        _block("number_1", 770, 40, 870, "112.", lines=1, of=36, kind=BoxKind.NUMBER),
        _block("heading_1", 230, 90, 760, "Protocoll, hållit wid lagtima", lines=1, of=36, kind=BoxKind.HEADING),
        _block("heading_2", 230, 120, 760, "Hösttinget den 3 October 1764.", lines=1, of=36, kind=BoxKind.HEADING),
        _block("note_1", 30, 200, 190, "§ 4.", lines=1, of=36, kind=BoxKind.NOTE),
        _block(
            "text_1",
            210,
            200,
            870,
            "Samma dag företogs det ifrån förra Tinget upskutna målet emellan Bonden Olof Ersson i Wästerby, "
            "kärande, och dess granne Per Jönsson därsammastädes, swarande, angående en gärdesgård, som den senare "
            "skall hafwa nederrifwit, hwarigenom kärandens åker af kreatur blifwit skadad.\n"
            "Parterna inställte sig personligen, och uplästes då den af Nämndemannen Lars Nilsson hållna synen, "
            "hwaruti skadan til twå tunnor korn wärderas.\n"
            "Swaranden nekade icke til gärningen, men föregaf, at gärdesgården stådt på hans egen mark.",
            lines=10,
            of=36,
        ),
        _block("note_2", 30, 510, 190, "Upskjutit.", lines=1, of=36, kind=BoxKind.NOTE),
        _block(
            "text_2",
            210,
            510,
            870,
            "Som Rätten fant nödigt at om rågången närmare underrättelse inhämta, ty upskjöts målet til nästa Ting, "
            "då Landtmätaren bör wara tilstädes med sin charta.",
            lines=3,
            of=36,
        ),
    ],
)

REGISTER = Document(
    aspect=0.707,
    lines=32,
    boxes=[
        _block(
            "heading_1",
            140,
            70,
            860,
            "Förtekning på Socknemagazinets Spanmål 1812.",
            lines=1,
            of=32,
            kind=BoxKind.HEADING,
        ),
        _rule("rule_1", 100, 120, 870, 124),
        _block(
            "text_1",
            110,
            140,
            580,
            "Anders Persson i Backa\nEnkan Brita Olsdotter\nNämndemannen Lars Larsson\nJonas Eriksson i Sörgården\n"
            "Torparen Nils Månsson\nOlof Andersson i Norrby\nKlockaren Petter Sundin\nSoldaten Carl Frisk\n"
            "Måns Jonsson i Hagen\nErik Olofsson i Wästergården",
            lines=10,
            of=32,
        ),
        _rule("rule_2", 598, 130, 602, 470),
        _block(
            "text_2",
            620,
            140,
            870,
            "3 Tunnor 4 Kappar\n1 Tunna 16 Kappar\n4 Tunnor\n2 Tunnor 8 Kappar\n20 Kappar\n3 Tunnor\n"
            "1 Tunna 4 Kappar\n12 Kappar\n2 Tunnor\n5 Tunnor 6 Kappar",
            lines=10,
            of=32,
        ),
        _rule("rule_3", 100, 470, 870, 474),
        _block("text_3", 110, 490, 580, "Summa", lines=1, of=32),
        _block("text_4", 620, 490, 870, "24 Tunnor 6 Kappar", lines=1, of=32),
        _block(
            "text_5",
            110,
            600,
            870,
            "At ofwanstående Spanmål blifwit uti Magazinet inlefwererad, warder härmed intygat.\n"
            "Sockenstugan den 20 December 1812.",
            lines=3,
            of=32,
        ),
        _block("text_6", 530, 740, 870, "Joh. Wallin, Pastor.", lines=1, of=32),
    ],
)

CERTIFICATE = Document(
    aspect=0.707,
    lines=28,
    boxes=[
        _block("heading_1", 300, 70, 700, "Prest-Betyg.", lines=1, of=28, kind=BoxKind.HEADING),
        _block("text_1", 110, 190, 290, "Namn:", lines=1, of=28),
        _block("text_2", 310, 190, 860, "Drängen Carl Johan Berg", lines=1, of=28),
        _block("text_3", 110, 270, 290, "Född den", lines=1, of=28),
        _block("text_4", 310, 270, 860, "6 Martii 1821 i Wåla Socken", lines=1, of=28),
        _block("text_5", 110, 350, 290, "Flyttar til", lines=1, of=28),
        _block("text_6", 310, 350, 860, "Upsala Domkyrko-Församling", lines=1, of=28),
        _block("text_7", 110, 430, 420, "Christendoms-kunskap:", lines=1, of=28),
        _block("text_8", 440, 430, 860, "godkänd, läser wäl innantil", lines=1, of=28),
        _block(
            "text_9",
            110,
            540,
            860,
            "Har til frägd och lefwerne mig weterligen fördt sig ärligen och wäl, samt är til ägtenskap ledig.",
            lines=2,
            of=28,
        ),
        _block("text_10", 110, 700, 660, "Wåla Prestgård den 2 November 1843.", lines=1, of=28),
        _block("text_11", 480, 780, 860, "E. Norberg, Comminister.", lines=1, of=28),
    ],
)

BLANK = Document(aspect=0.707, lines=30, boxes=[])

TEMPLATES: dict[str, Document] = {
    "Letter": LETTER,
    "Court minutes": MINUTES,
    "Register": REGISTER,
    "Certificate": CERTIFICATE,
    "Blank page": BLANK,
}
