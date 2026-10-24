"""Where a page comes from, as a caption names it: the archive (collection) and the volume's years."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

YEARS = re.compile(r"(1[5-9]\d\d)(?:\s*[-–]\s*(1[5-9]\d\d))?(-tal|-luku)?")
# Folder names that went through a Mac SMB share carry the "SFM" encoding of characters Windows forbids at the end of
# a name: a trailing space as U+F028, a trailing period as U+F029 (two GT collection folders end in U+F028).
_SFM_DECODE = str.maketrans({"\uf028": " ", "\uf029": "."})


# The GT collection folder (minus "GT-" and the export date) -> the name its captions use: the style handle
# that picks a collection's hand, paper and ink at generation time.
COLLECTION_TITLES = {
    "FNA_1500": "the 16th-century records of the National Archives of Finland",
    "FNA_finnish_court_records": "Finnish court records in the National Archives of Finland",
    "FNA_finnish_letter_recipes": "a Finnish book of letters and recipes",
    "JLF_jamtlands_domsaga": "the court records of Jämtlands domsaga",
    "RA_1634_ars_regeringsform": "the 1634 Instrument of Government (1634 års regeringsform)",
    "RA_alvsborgs_losen": "the Älvsborgs lösen tax rolls",
    "RA_arjeplogs_tingslag": "the court minutes of Arjeplogs tingslag",
    "RA_bergmastare_i_nora": "the hammer-court minutes of Bergmästaren i Nora",
    "RA_bergskollegium_rattsprotokoll": "the court minutes of Bergskollegium",
    "RA_bergskollegium_relationer_och_skrivelser": "the reports and letters of Bergskollegium",
    "RA_bergskollegium_stora_kopparbergs_bergslag": "the Bergskollegium records of Stora Kopparbergs bergslag",
    "RA_boskapslangder": "the boskapslängder livestock tax rolls",
    "RA_carl_fredrik_pahlmans_resejournaler": "Carl Fredrik Påhlman's travel journals",
    "RA_frihetstidens_utskottshandlingar": "the committee papers of the Age of Liberty diets",
    "RA_gota_hovratt_brottsmalsprotokoll": "the criminal-case minutes of Göta hovrätt",
    "RA_goteborgs_poliskammare": "the records of Göteborgs poliskammare",
    "RA_harnosands_domkapitlet": "the records of Härnösands domkapitel",
    "RA_jonkopings_radhusratts_och_magistrat": "the records of Jönköpings rådhusrätt och magistrat",
    "RA_krigsarkivet_dombocker": "the regimental court records of Krigsarkivet",
    "RA_landskapshandlingar": "the landskapshandlingar provincial accounts",
    "RA_landtmatarnes_ankomne_ordres": "the land surveyors' received orders (Landtmätarnes ankomne ordres)",
    "RA_riksregistraturet": "Riksregistraturet, the royal chancery register",
    "RA_strangnas_domkapitel": "the records of Strängnäs domkapitel",
    "RA_strodda_kamerala_handlingar": "the strödda kamerala handlingar fiscal records",
    "RA_svea_hovratt_koncept": "the draft minutes of Svea hovrätt",
    "RA_svea_hovratt_renskrivna_protokoll": "the fair-copy minutes of Svea hovrätt",
    "RA_trolldomskommissionen": "the records of the witchcraft commissions (trolldomskommissionen)",
    "RA_uppvidinge_haradsratt": "the court records of Uppvidinge häradsrätt",
    "RA_vase_haradsratt": "the court records of Väse häradsrätt",
    "RA-X_beskrivning_over_provinsen_gastrikland": "a description of the province of Gästrikland",
    "RA-X_gbg_1790-1794": "a Göteborg register",
    "RA_yngre_geometriska_kartor": "the younger geometric maps (yngre geometriska kartor)",
    "SSA_allmanna_barnhuset": "the records of Allmänna barnhuset in Stockholm",
    "SSA_borgmastare_och_rad_fore_1636": "the minutes of Stockholm's borgmästare och råd",
    "SSA_botkyrka_kyrkoarkiv": "the parish records of Botkyrka",
    "SSA_brannkyrka_kyrkoarkiv": "the parish records of Brännkyrka",
    "SSA_byggningskollegium": "the records of Stockholm's byggningskollegium",
    "SSA_deputerade": "the records of Stockholm's deputerade",
    "SSA_fritjof_englund": "Fritjof Englund's papers",
    "SSA_grödinge_kyrkoarkiv": "the parish records of Grödinge",
    "SSA_hall-_och_manufakturratten": "the records of Hall- och manufakturrätten",
    "SSA_handelskollegiet": "the records of Stockholm's handelskollegium",
    "SSA_justitiekollegiet_och_formyndarkammaren": "the records of Justitiekollegiet och förmyndarkammaren",
    "SSA_kungliga_hovkonsistoriet": "the records of Kungliga hovkonsistoriet",
    "SSA_norra_forstadens_kamnarsratt": "the court records of Norra förstadens kämnärsrätt",
    "SSA_norra_forstadens_vastra_kamnarsratt": "the court records of Norra förstadens västra kämnärsrätt",
    "SSA_overstathallarambetets_aldre_kansli": "the records of Överståthållarämbetet",
    "SSA_politikollegiet": "the records of Stockholm's politikollegium",
    "SSA_sodra_forstadens_kamnarsratt": "the court records of Södra förstadens kämnärsrätt",
    "SSA_sollentuna_haradsratt": "the court records of Sollentuna häradsrätt",
    "SSA_sotholms_haradsratt": "the court records of Sotholms häradsrätt",
    "SSA_spånga_kyrkoarkiv": "the parish records of Spånga",
    "SSA_stadens_kamnarsratt": "the court records of Stockholm's stadens kämnärsrätt",
    "SSA_stockholm_magistrat_och_radhusratt": "the records of Stockholms magistrat och rådhusrätt",
    "SSA_stockholms_domkapitel": "the records of Stockholms domkapitel",
    "SSA_svartlosa_haradsratt": "the court records of Svartlösa häradsrätt",
    "VKaariainen_1600-luku_GT": "17th-century Finnish records",
}


class Provenance(BaseModel):
    """Where a crop comes from, as its caption names it."""

    model_config = {"frozen": True}

    title: str  # the collection
    years: str  # the volume's, '' when its name has none


def provenance_of(collection: str, page_xml: Path) -> Provenance:
    """The archive and years a caption names for a page: ``<collection>/<volume>/<volume>/page/<file>.xml``."""
    return Provenance(title=collection_title(collection), years=volume_years(page_xml.parent.parent.name))


def collection_key(collection: str) -> str:
    """A GT collection folder's name without the ``GT-`` prefix and the export date: its key in COLLECTION_TITLES.

    ``GT-JLF_jamtlands_domsaga_20260220`` -> ``JLF_jamtlands_domsaga``.
    """
    return re.sub(r"^GT-|_\d{8}$", "", collection.translate(_SFM_DECODE).strip())


def collection_title(collection: str) -> str:
    """The caption name of a GT collection folder (its slug for a collection COLLECTION_TITLES lacks)."""
    key = collection_key(collection)
    return COLLECTION_TITLES.get(key, key.replace("_", " "))


def volume_years(volume: str) -> str:
    """The years in a volume folder's name ('1676–1677', '1777', '1690s'), preferring a range; '' if none."""
    found = list(YEARS.finditer(volume))
    if not found:
        return ""
    ranges = [match for match in found if match.group(2)]
    first, last, century_suffix = (ranges or found)[0].groups()
    if last:
        return f"{first}–{last}"
    if century_suffix:
        return f"{first}s"
    return first
