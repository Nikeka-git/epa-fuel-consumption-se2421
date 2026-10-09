"""Enrich observed raw-field inventory from the captured official API guide, offline.

The original guide and raw responses are never changed. Undocumented encodings
remain explicitly unresolved rather than being assigned guessed meanings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from fuel_consumption.utils import project_root, sha256_file, write_json


class ListDescriptions(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.descriptions = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'li':
            if self.stack:
                self.stack[-1]['frozen'] = True
            self.stack.append({'text': [], 'frozen': False})
        elif tag == 'div' and self.stack:
            self.stack[-1]['frozen'] = True

    def handle_data(self, text):
        if self.stack and not self.stack[-1]['frozen']:
            self.stack[-1]['text'].append(text)

    def handle_endtag(self, tag):
        if tag == 'li' and self.stack:
            text = re.sub(r'\s+', ' ', ''.join(self.stack.pop()['text'])).strip()
            match = re.match(r'^([A-Za-z][A-Za-z0-9_]*)\s*-\s*(.+)', text)
            if match:
                key, meaning = match.groups()
                self.descriptions.setdefault(key, meaning)


# The guide uses lowercase prose aliases, while individual JSON uses camelCase.
ALIASES = {'atvType': 'atvtype', 'baseModel': 'basemodel'}
EXPLICIT = {
    'emissionsList': ('Nested emissions records linked to this vehicle; see the emissions section of the API guide.', 'documented_source_structure'),
    'evMotor': ('Electric-motor source description. The guide labels it as kW, but actual responses can contain free-form power, voltage and battery text; preserve it as a string.', 'documented_and_observed_type_review'),
    'battery': ('Battery-related source field. Its exact meaning and encoding are absent from the captured guide; retained without interpretation and excluded from inputs.', 'undocumented_source_encoding'),
    'cylDeact': ('Cylinder-deactivation source code inferred from the field name; exact code mapping is not documented in the captured API guide.', 'field_name_inference_encoding_unconfirmed'),
    'cylDeactYesNo': ('Human-readable cylinder-deactivation indicator inferred from the field name; not used for eligibility or prediction.', 'field_name_inference_encoding_unconfirmed'),
    'mpgRevised': ('Source flag apparently indicating revised MPG estimates; exact field encoding is absent from the captured guide. Not a predictor.', 'field_name_inference_encoding_unconfirmed'),
    'range': ('Driving-range source field for the primary fuel. Exact encoding and units are not defined for this key in the captured API guide; excluded from inputs.', 'undocumented_source_encoding'),
    'rangeCity': ('City driving-range source field for the primary fuel. Exact encoding and units are not defined for this key in the captured guide; excluded from inputs.', 'undocumented_source_encoding'),
    'rangeHwy': ('Highway driving-range source field for the primary fuel. Exact encoding and units are not defined for this key in the captured guide; excluded from inputs.', 'undocumented_source_encoding'),
}


def enrich(interim: Path, documentation: Path) -> dict:
    source = interim / 'raw_field_dictionary.json'
    inventory = json.loads(source.read_text(encoding='utf-8'))
    parser = ListDescriptions()
    parser.feed(documentation.read_text(encoding='utf-8'))
    for key, entry in inventory.items():
        documented_key = ALIASES.get(key, key)
        if key in EXPLICIT:
            meaning, status = EXPLICIT[key]
        elif documented_key in parser.descriptions:
            meaning = parser.descriptions[documented_key]
            status = 'documented_official_api_guide'
        else:
            meaning = 'Undocumented source field; its meaning is unresolved and it is excluded from model inputs.'
            status = 'undocumented_source_encoding'
        entry.update(description=meaning, meaning_review_status=status,
                     description_source='https://www.fueleconomy.gov/feg/ws/index.shtml',
                     description_source_sha256=sha256_file(documentation),
                     documented_key=documented_key)
    write_json(source, inventory)
    fields = ['raw_field', 'internal_field', 'description', 'meaning_review_status',
              'disposition', 'present_count', 'missing_or_empty_count', 'observed_json_types',
              'description_source', 'documented_key']
    with (interim / 'raw_field_dictionary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, entry in inventory.items():
            row = {'raw_field': key, **{field: entry.get(field) for field in fields[1:]}}
            row['observed_json_types'] = json.dumps(row['observed_json_types'], sort_keys=True)
            writer.writerow(row)
    result = {'fields': len(inventory), 'review_status_counts': dict(Counter(e['meaning_review_status'] for e in inventory.values())),
              'unconfirmed_keys': [key for key, e in inventory.items() if 'unconfirmed' in e['meaning_review_status'] or 'undocumented' in e['meaning_review_status']],
              'unconfirmed_inputs': [key for key, e in inventory.items() if e['disposition'] == 'structured_predictor' and ('unconfirmed' in e['meaning_review_status'] or 'undocumented' in e['meaning_review_status'])]}
    write_json(interim / 'dictionary_review.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interim', type=Path, default=Path('data/interim'))
    args = parser.parse_args()
    print(json.dumps(enrich(args.interim, project_root() / 'evidence/source_policy/api_documentation.html'), indent=2))


if __name__ == '__main__':
    main()
