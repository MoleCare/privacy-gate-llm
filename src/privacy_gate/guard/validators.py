# SPDX-License-Identifier: Apache-2.0
"""Validators: the checks a regex cannot make.

A pattern finds the *shape* of a card number, an IBAN or an NHS number. Most strings of that shape are not
one: a 16-digit order id, a 22-character reference, ten digits that happen to be spaced 3-3-4. The checksum
is what turns a shape into a finding, and it is why these categories can be blocked without blocking every
long number in a log.

Every validator takes the span text exactly as matched and returns a bool. They are pure, standard library
only, and named in `rules/privacy-v2.yaml`; a rule naming a validator this module does not have fails to
load, so a TypeScript engine that forgets one fails its conformance run rather than silently matching more.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Callable

# IBAN length per country, from the SWIFT IBAN registry. A country missing here makes the IBAN invalid, which
# is the safe direction for a *mask* category: a miss is reported by the conformance suite when someone adds
# a fixture, and adding a country is a one-line change.
IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22, "BR": 29,
    "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29,
    "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28,
    "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20,
    "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19,
    "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29,
    "RO": 24, "RS": 22, "SA": 24, "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}


def digits(text: str) -> str:
    return "".join(ch for ch in text if "0" <= ch <= "9")


def luhn(text: str) -> bool:
    """Payment card: 13 to 19 digits whose Luhn sum is a multiple of ten."""
    number = digits(text)
    if not 13 <= len(number) <= 19:
        return False
    total = 0
    for position, ch in enumerate(reversed(number)):
        value = ord(ch) - 48
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_mod97(text: str) -> bool:
    """IBAN: known country, that country's length, and ISO 7064 mod 97-10 equal to 1."""
    compact = "".join(text.split()).upper()
    if len(compact) < 5 or not compact.isascii() or not compact.isalnum():
        return False
    if IBAN_LENGTHS.get(compact[:2]) != len(compact):
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(number) % 97 == 1


def nhs_mod11(text: str) -> bool:
    """NHS number (England, Wales, Isle of Man): ten digits, weights 10 to 2, check digit by modulus 11.
    A remainder giving 10 means the number was never issued."""
    number = digits(text)
    if len(number) != 10:
        return False
    total = sum((10 - i) * (ord(ch) - 48) for i, ch in enumerate(number[:9]))
    check = 11 - total % 11
    if check == 11:
        check = 0
    if check == 10:
        return False
    return check == ord(number[9]) - 48


def us_ssn_area(text: str) -> bool:
    """US Social Security number in AAA-GG-SSSS form. There is no checksum; the SSA's issuing rules are
    what is checkable: area not 000, 666 or 900-999, group not 00, serial not 0000."""
    number = digits(text)
    if len(number) != 9:
        return False
    area, group, serial = int(number[:3]), int(number[3:5]), int(number[5:])
    if area == 0 or area == 666 or area >= 900:
        return False
    return group != 0 and serial != 0


_PRIVATE_V4 = tuple(ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",  # RFC 1918
    "169.254.0.0/16",                                 # link-local, RFC 3927
    "100.64.0.0/10",                                  # carrier-grade NAT, RFC 6598
))
_PRIVATE_V6 = tuple(ipaddress.ip_network(n) for n in (
    "fc00::/7",   # unique local, RFC 4193
    "fe80::/10",  # link-local
))


def private_ipv4(text: str) -> bool:
    """An address inside a private, link-local or CGNAT range. A last octet of 0 is a network address
    (`10.0.0.0/8` in a firewall rule or a README), which names a range, not a machine, so it is not held."""
    try:
        address = ipaddress.IPv4Address(text)
    except ValueError:
        return False
    if text.rsplit(".", 1)[-1] == "0":
        return False
    return any(address in network for network in _PRIVATE_V4)


def private_ipv6(text: str) -> bool:
    """A unique-local or link-local IPv6 address. A zone index (`%en0`) is not part of the match."""
    try:
        address = ipaddress.IPv6Address(text)
    except ValueError:
        return False
    return any(address in network for network in _PRIVATE_V6)


def phone_digits(text: str) -> bool:
    """A telephone number: 10 to 15 digits (E.164 allows at most 15), and not one digit repeated, which is a
    placeholder (`000 000 0000`) rather than a number."""
    number = digits(text)
    if not 10 <= len(number) <= 15:
        return False
    return len(set(number)) > 1


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "luhn": luhn,
    "iban_mod97": iban_mod97,
    "nhs_mod11": nhs_mod11,
    "us_ssn_area": us_ssn_area,
    "private_ipv4": private_ipv4,
    "private_ipv6": private_ipv6,
    "phone_digits": phone_digits,
}
