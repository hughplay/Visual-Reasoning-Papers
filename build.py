import argparse
import os
from datetime import datetime

import bibtexparser
import rebiber
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import splitname


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
            if "Image Process" in entry[key]:
                return "TIP"
            if "Association for Computational Linguistics" in entry[key]:
                return "ACL"

    if "url" in entry and "arxiv" in entry["url"]:
        return "arXiv"
    return ""


def rebiber_bib(raw_datasets_bib, raw_methods_bib, output_bib):
    dataset_entries = rebiber.load_bib_file(raw_datasets_bib)
    methods_entries = rebiber.load_bib_file(raw_methods_bib)
    all_bib_entries = dataset_entries + methods_entries
    filepath = os.path.abspath(rebiber.__file__).replace("__init__.py", "")
    bib_list_path = os.path.join(filepath, "bib_list.txt")
    bib_db = rebiber.construct_bib_db(bib_list_path, start_dir=filepath)
    rebiber.normalize_bib(bib_db, all_bib_entries, output_bib, sort=True)


def render_markdown(input_md, output_md, entries, dataset_keys):
    mentioned_keys = []
    rendered_lines = []
    with open(input_md) as template_file:
        lines = template_file.readlines()

    render_on = False
    for line in lines:

        if "Last update time" in line:
            line = line.replace("{date}", datetime.now().strftime("%Y-%m-%d"))

        if "{paper_list_by_year}" in line:
            line = ""
            year = None
            for entry in sorted(
                entries.values(), key=lambda x: x["year"], reverse=True
            ):
                if year != entry["year"]:
                    year = entry["year"]
                    line += f"\n## {year}\n"
                line += render_paper(
                    entry, is_dataset=entry["ID"] in dataset_keys
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
                line = render_paper(entry, is_dataset=entry_key in dataset_keys)

        rendered_lines.append(line)

    with open(output_md, "w") as output_file:
        output_file.writelines(rendered_lines)

    return mentioned_keys


def render_paper(entry, is_dataset=False):
    line = "-"

    if is_dataset:
        line += " ★"

    title = entry["title"].replace("{", "").replace("}", "")
    line += f" **{title}**"

    if "author" in entry:
        authors = [
            splitname(name.strip()) for name in entry["author"].split("and")
        ]
        if len(authors) > 2:
            line += f", {authors[0]['last'][0]} et al."
        else:
            line += f", {' & '.join([author['last'][0] for author in authors])}"

    abbr = conference_abbr(entry)
    year = entry["year"] if "year" in entry else ""

    line += f", *{abbr} {year}*"

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
    return line


def main(raw_datasets_bib, raw_methods_bib, output_bib, input_md, output_md):

    print("Rebibering bib files...")
    rebiber_bib(raw_datasets_bib, raw_methods_bib, output_bib)

    print("Reading dataset keys...")
    with open(raw_datasets_bib) as f_bib:
        parser = BibTexParser()
        dataset_keys = list(
            bibtexparser.load(f_bib, parser=parser).entries_dict.keys()
        )

    print("Reading bib file...")
    with open(output_bib) as f_bib:
        parser = BibTexParser()
        entries = bibtexparser.load(f_bib, parser=parser).entries_dict

    print("Rendering markdown...")
    render_markdown(input_md, output_md, entries, dataset_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_datasets_bib", type=str, default="raw_datasets.bib"
    )
    parser.add_argument(
        "--raw_methods_bib", type=str, default="raw_methods.bib"
    )
    parser.add_argument(
        "--output_bib", type=str, default="visual_reasoning.bib"
    )
    parser.add_argument("--input_md", type=str, default="TEMPLATE.md")
    parser.add_argument("--output_md", type=str, default="README.md")
    args = parser.parse_args()

    main(
        args.raw_datasets_bib,
        args.raw_methods_bib,
        args.output_bib,
        args.input_md,
        args.output_md,
    )
