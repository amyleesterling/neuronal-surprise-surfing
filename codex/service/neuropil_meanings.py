"""Plain-English role for each FlyWire neuropil code.

Used to write human-readable surprise descriptions instead of dumping raw
acronyms ("AMMC_R → GA_L") on the user. The role descriptions are deliberately
short (a single noun phrase) so they slot into headlines like:

    "AMMC_R (auditory) → GA_L (heading/spatial)"

Coverage is best-effort across the major FlyWire neuropil names. Missing
codes fall back to the bare acronym, which is still legible.

Sources cross-referenced from FlyWire annotations and the Drosophila brain
atlas literature (Ito et al. 2014; FlyWire Schlegel et al. 2024).
"""

# Map base neuropil code (without _L/_R suffix) → short role phrase.
NEUROPIL_ROLE = {
    # Visual system (input)
    "LA": "lamina (visual input layer)",
    "ME": "medulla (visual processing)",
    "LO": "lobula (visual feature integration)",
    "LOP": "lobula plate (motion detection)",
    "AOTU": "anterior optic tubercle (visual relay)",
    # Olfactory system
    "AL": "antennal lobe (smell)",
    "LH": "lateral horn (innate odor response)",
    # Mushroom body — learning & memory
    "MB_CA": "mushroom body calyx (memory input)",
    "MB_PED": "mushroom body peduncle (memory axon shaft)",
    "MB_ML": "mushroom body medial lobe (memory output)",
    "MB_VL": "mushroom body vertical lobe (memory output)",
    # Central complex — heading, navigation, sleep
    "EB": "ellipsoid body (heading)",
    "FB": "fan-shaped body (sleep & motor planning)",
    "PB": "protocerebral bridge (heading)",
    "NO": "noduli (heading)",
    "BU": "bulb (visual→heading relay)",
    "GA": "gall (heading-adjacent)",
    "LAL": "lateral accessory lobe (premotor)",
    # Auditory / mechanosensory
    "AMMC": "antennal mechanosensory & motor center (sound & touch)",
    "WED": "wedge (auditory & multimodal)",
    # Taste / feeding
    "GNG": "gnathal ganglion (taste & feeding motor)",
    "SAD": "saddle (gustatory)",
    "CAN": "cantle (gustatory-adjacent)",
    # Higher-order protocerebrum
    "AVLP": "ventrolateral protocerebrum (multisensory)",
    "PVLP": "posterior ventrolateral protocerebrum (multisensory)",
    "PLP": "posterior lateral protocerebrum (visual & multimodal)",
    "SLP": "superior lateral protocerebrum (higher-order)",
    "SIP": "superior intermediate protocerebrum (higher-order)",
    "SMP": "superior medial protocerebrum (higher-order)",
    "SCL": "superior clamp (associative)",
    "ICL": "inferior clamp (associative)",
    "SPS": "superior posterior slope",
    "IPS": "inferior posterior slope",
    "ATL": "antler (premotor-adjacent)",
    "CRE": "crepine (central)",
    "IB": "inferior bridge (central)",
    "EPA": "epaulette",
    "VES": "vest",
    "GOR": "gorget",
    "FLA": "flange",
    "PRW": "prow",
    "OCG": "ocellar ganglion (light detection)",
    "GC": "great commissure",
    "OPTU": "optic tubercle",
    # Catch-alls
    "UNASGD": "unassigned region",
    "OUTSIDE": "outside connectome",
}


def _strip_side(code: str) -> tuple[str, str]:
    """Split a neuropil code into (base, side_suffix) tuple.
    'AMMC_R' -> ('AMMC', ' (right)'); 'EB' -> ('EB', '')"""
    if not code:
        return "", ""
    if code.endswith("_L"):
        return code[:-2], " (left)"
    if code.endswith("_R"):
        return code[:-2], " (right)"
    return code, ""


def role(code: str) -> str:
    """Return a short prose role for a neuropil code, or the bare code if
    unknown. Examples:
        role('AMMC_R')   -> 'antennal mechanosensory & motor center (sound & touch), right'
        role('MB_CA_L')  -> 'mushroom body calyx (memory input), left'
        role('XYZ_R')    -> 'XYZ (right)'
    """
    if not code:
        return ""
    base, side = _strip_side(code)
    role_phrase = NEUROPIL_ROLE.get(base)
    if role_phrase:
        if side:
            # 'antennal lobe (smell)' + ' (right)' reads weirdly; merge:
            return f"{role_phrase}, {side.strip(' ()')}"
        return role_phrase
    # Unknown base → just the original code, with side spelled out if any
    return code if not side else f"{base}{side}"


def short_role(code: str) -> str:
    """Like role(), but without the side suffix — for compact headlines."""
    if not code:
        return ""
    base, _ = _strip_side(code)
    return NEUROPIL_ROLE.get(base, base)
