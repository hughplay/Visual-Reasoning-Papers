import argparse
from datetime import datetime

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import *
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase


def conference_abbr(entry):
    for key in ["booktitle", "journal"]:
        if key in entry:
            if "CVPR" in entry[key]:
                return "CVPR"
            if "Neural Inf" in entry[key]:
                return "NeurIPS"
            if "ICCV" in entry[key]:
                return "ICCV"
            if "ECCV" in entry[key]:
                return "ECCV"
            if "ICML" in entry[key]:
                return "ICML"
            if "IJCAI" in entry[key]:
                return "IJCAI"
            if "AAAI" in entry[key]:
                return "AAAI"
            if "Pattern Anal" in entry[key]:
                return "TPAMI"
            if "Learning Representations" in entry[key]:
                return "ICLR"

    if "url" in entry and "arxiv" in entry["url"]:
        return "arXiv"
    return ""


def main(args):
    with open(args.input_bib) as input_bib:
        parser = BibTexParser()
        entries = bibtexparser.load(input_bib, parser=parser).entries_dict

    mentioned_keys = render_markdown(args.input_md, args.output_md, entries)
    mentioned_entries = []
    for key in mentioned_keys:
        mentioned_entries.append(entries[key])

    db = BibDatabase()
    db.entries = mentioned_entries

    writer = BibTexWriter()
    writer.indent = "  "
    with open(args.output_bib, "w") as bibfile:
        bibfile.write(writer.write(db))


def render_markdown(input_md, output_md, entries):
    mentioned_keys = []
    rendered_lines = []
    with open(input_md) as template_file:
        lines = template_file.readlines()

    render_on = False
    for line in lines:

        if "Last update time" in line:
            line = line.replace(
                "{date}", datetime.now().strftime("%Y-%m-%d")
            )

        if line.startswith("<!-- BEGIN ENTRIES -->"):
            render_on = True
        elif line.startswith("<!-- END ENTRIES -->"):
            render_on = False

        if render_on:
            entry_key = line.strip().split(" ")[-1]
            if entry_key in entries:
                entry = entries[entry_key]
                mentioned_keys.append(entry_key)

                title = entry["title"].replace("{", "").replace("}", "")
                line = f"- **{title}**"

                if "author" in entry:
                    if len(entry["author"].split(" and ")) > 1:
                        line += f", {entry['author'].split(' and ')[0].split(',')[-1]} et al."
                    else:
                        line += f", {entry['author'].replace(',', '')}"

                abbr = conference_abbr(entry)
                year = entry["year"] if "year" in entry else ""

                line += f", {abbr} {year}"

                line += "."

                if "url" in entry:
                    line += f" [Paper]({entry['url']})"

                if "project" in entry:
                    if not line.endswith("."):
                        line += " / "
                    line += f" [Project]({entry['project']})"

                if "code" in entry:
                    if not line.endswith("."):
                        line += " / "
                    line += f" [Code]({entry['code']})"

                line += "\n"
        rendered_lines.append(line)

    with open(output_md, "w") as output_file:
        output_file.writelines(rendered_lines)

    return mentioned_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_bib", type=str, default="visual_reasoning_raw.bib"
    )
    parser.add_argument(
        "--output_bib", type=str, default="visual_reasoning.bib"
    )
    parser.add_argument("--input_md", type=str, default="TEMPLATE.md")
    parser.add_argument("--output_md", type=str, default="README.md")
    args = parser.parse_args()
    main(args)
