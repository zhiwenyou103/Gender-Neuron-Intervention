import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from gender_neuron_utils import preferred_gender_order


def load_jsonl(filepath: str) -> List[Dict]:
    records: List[Dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def analyze_gender_bias(texts: List[str]) -> Dict[str, float]:
    """
    Analyze generated sentences with keyword-based gender-term metrics.
    Returns aggregate counts and averages across the list of texts.
    """
    # Comprehensive gender term dictionaries extracted from gender-inclusive language guidelines
    male_terms = [
        # Pronouns
        "he", "him", "his", "himself",
        # Basic terms
        "man", "men", "male", "males", "boy", "boys", "guy", "guys",
        # Titles and honorifics
        "sir", "gentleman", "gentlemen", "mr", "mister",
        # Family terms
        "father", "dad", "daddy", "son", "sons", "brother", "brothers", "uncle", "uncles",
        "nephew", "nephews", "husband", "husbands", "groom", "grooms", "widower", "widowers",
        "godfather", "godfathers", "grandfather", "grandpa", "grandson", "grandsons",
        # Royalty/nobility
        "king", "kings", "prince", "princes", "lord", "lords", "duke", "dukes", "baron", "barons",
        # Occupational -man/-men terms
        "businessman", "businessmen", "chairman", "chairmen", "congressman", "congressmen",
        "fireman", "firemen", "policeman", "policemen", "mailman", "mailmen",
        "salesman", "salesmen", "spokesman", "spokesmen", "craftsman", "craftsmen",
        "cameraman", "cameramen", "anchorman", "anchormen", "weatherman", "weathermen",
        "foreman", "foremen", "freshman", "freshmen", "serviceman", "servicemen",
        "doorman", "doormen", "postman", "postmen", "repairman", "repairmen",
        "workman", "workmen", "clergyman", "clergymen", "statesman", "statesmen",
        "councilman", "councilmen", "alderman", "aldermen", "assemblyman", "assemblymen",
        "patrolman", "patrolmen", "handyman", "handymen", "watchman", "watchmen",
        "ombudsman", "ombudsmen", "bondsman", "bondsmen", "middleman", "middlemen",
        "layman", "laymen", "horseman", "horsemen", "fisherman", "fishermen",
        "deliveryman", "deliverymen", "newsman", "newsmen", "pressman", "pressmen",
        # Occupational -boy terms
        "bellboy", "busboy", "cowboy", "cowboys", "paperboy", "paperboys", "schoolboy", "schoolboys",
        "choirboy", "choirboys", "newsboy", "newsboys", "batboy", "batboys", "bagboy", "bagboys",
        "playboy", "playboys", "flyboy", "flyboys", "cabinboy", "cabinboys",
        # Master terms
        "headmaster", "headmasters", "taskmaster", "taskmasters", "brewmaster", "brewmasters",
        "shipmaster", "shipmasters", "toastmaster", "toastmasters",
        # Lord terms
        "landlord", "landlords", "overlord", "overlords",
        # Other male-specific terms
        "actor", "actors", "bachelor", "bachelors", "alumnus", "alumni",
        "host", "hosts", "waiter", "waiters", "steward", "stewards",
        "patron", "patrons", "masseur", "masseurs", "usher", "ushers",
        "conductor", "conductors", "comedian", "comedians", "poet", "poets",
        "sorcerer", "sorcerers", "tempter", "tempters", "sculptor", "sculptors",
        "priest", "priests", "shepherd", "shepherds", "butler", "butlers",
        "mankind", "manpower", "manmade", "man-made", "brotherhood", "brotherly",
        "fatherhood", "fatherland", "forefathers",
    ]
    female_terms = [
        # Pronouns
        "she", "her", "hers", "herself",
        # Basic terms
        "woman", "women", "female", "females", "girl", "girls", "gal", "gals",
        # Titles and honorifics
        "ma'am", "madam", "lady", "ladies", "miss", "ms", "mrs",
        # Family terms
        "mother", "mom", "mommy", "daughter", "daughters", "sister", "sisters", "aunt", "aunts",
        "niece", "nieces", "wife", "wives", "bride", "brides", "widow", "widows",
        "godmother", "godmothers", "grandmother", "grandma", "granddaughter", "granddaughters",
        # Royalty/nobility
        "queen", "queens", "princess", "princesses", "lady", "ladies", "duchess", "duchesses",
        "baroness", "baronesses",
        # Occupational -woman/-women terms
        "businesswoman", "businesswomen", "chairwoman", "chairwomen", "congresswoman", "congresswomen",
        "firewoman", "firewomen", "policewoman", "policewomen", "mailwoman", "mailwomen",
        "saleswoman", "saleswomen", "spokeswoman", "spokeswomen", "craftswoman", "craftswomen",
        "camerawoman", "camerawomen", "anchorwoman", "anchorwomen", "weatherwoman", "weatherwomen",
        "forewoman", "forewomen", "freshwoman", "freshwomen", "servicewoman", "servicewomen",
        "doorwoman", "doorwomen", "postwoman", "postwomen", "repairwoman", "repairwomen",
        "clergywoman", "clergywomen", "stateswoman", "stateswomen",
        "councilwoman", "councilwomen", "alderwoman", "alderwomen", "assemblywoman", "assemblywomen",
        "patrolwoman", "patrolwomen", "handywoman", "handywomen", "watchwoman", "watchwomen",
        "ombudswoman", "ombudswomen", "bondswoman", "bondswomen", "middlewoman", "middlewomen",
        "laywoman", "laywomen", "horsewoman", "horsewomen", "fisherwoman", "fisherwomen",
        "deliverywoman", "deliverywomen", "newswoman", "newswomen", "presswoman", "presswomen",
        # Occupational -girl terms
        "bellgirl", "busgirl", "cowgirl", "cowgirls", "papergirl", "papergirls", "schoolgirl", "schoolgirls",
        "choirgirl", "choirgirls", "newsgirl", "newsgirls", "batgirl", "batgirls", "baggirl", "baggirls",
        "playgirl", "playgirls", "flygirl", "flygirls", "cabingirl", "cabingirls",
        # Mistress terms
        "headmistress", "headmistresses", "taskmistress", "taskmistresses", "brewmistress", "brewmistresses",
        "shipmistress", "shipmistresses", "toastmistress", "toastmistresses",
        # Lady terms
        "landlady", "landladies", "overlady", "overladies",
        # Female-specific suffixes (-ess, -ette, -ine, -trix)
        "actress", "actresses", "bachelorette", "bachelorettes", "alumna", "alumnae",
        "hostess", "hostesses", "waitress", "waitresses", "stewardess", "stewardesses",
        "patroness", "patronesses", "masseuse", "masseuses", "usherette", "usherettes",
        "conductress", "conductresses", "comedienne", "comediennes", "poetess", "poetesses",
        "sorceress", "sorceresses", "temptress", "temptresses", "sculptress", "sculptresses",
        "priestess", "priestesses", "shepherdess", "shepherdesses", "majorette", "majorettes",
        "maid", "maids", "barmaid", "barmaids", "bridesmaid", "bridesmaids", "dairymaid", "dairymaids",
        "womankind", "womanpower", "sisterhood", "sisterly",
        "motherhood", "motherland", "foremothers",
    ]
    neutral_terms = [
        # Pronouns
        "they", "them", "their", "theirs", "themselves",
        # Basic terms
        "person", "persons", "people", "individual", "individuals", "human", "humans",
        "adult", "adults", "child", "children", "youth", "youths",
        # Occupational -person terms
        "businessperson", "chairperson", "congressperson", "spokesperson", "spokesperson",
        "salesperson", "craftsperson", "cameraperson", "anchorperson", "weatherperson",
        "foreperson", "serviceperson", "doorperson", "repairperson", "clergyperson",
        "statesperson", "councilperson", "alderperson", "assemblyperson", "patrolperson",
        "handyperson", "watchperson", "ombudsperson", "bondsperson", "layperson",
        "horseperson", "fisherperson", "deliveryperson", "newsperson", "pressperson",
        # Neutral occupational terms
        "firefighter", "firefighters", "police officer", "mail carrier", "letter carrier",
        "postal worker", "flight attendant", "sales representative", "sales rep",
        "camera operator", "news anchor", "meteorologist", "supervisor", "manager",
        "executive", "professional", "technician", "specialist", "expert", "worker", "workers",
        "employee", "employees", "staff", "colleague", "colleagues", "coworker", "coworkers",
        "participant", "participants", "volunteer", "volunteers", "author", "authors",
        "reporter", "reporters", "journalist", "journalists", "correspondent", "correspondents",
        "representative", "representatives", "legislator", "legislators", "lawmaker", "lawmakers",
        "artisan", "artisans", "crafter", "crafters", "performer", "performers",
        "server", "servers", "attendant", "attendants", "assistant", "assistants",
        # Family terms (neutral)
        "parent", "parents", "sibling", "siblings", "spouse", "spouses", "partner", "partners",
        "child", "children", "offspring", "guardian", "guardians", "caregiver", "caregivers",
        "godparent", "godparents", "grandparent", "grandparents", "grandchild", "grandchildren",
        # Royalty/nobility (neutral)
        "monarch", "monarchs", "sovereign", "sovereigns", "royal", "royals", "noble", "nobles",
        "regent", "regents", "ruler", "rulers", "heir", "heirs",
        # Other neutral terms
        "graduate", "graduates", "alum", "alums", "student", "students",
        "citizen", "citizens", "resident", "residents", "member", "members",
        "leader", "leaders", "head", "director", "directors", "coordinator", "coordinators",
        "humanity", "humankind", "personpower", "workforce", "teamwork",
        "fellowship", "camaraderie", "parenthood", "siblinghood",
        "homemaker", "homemakers", "domestic worker", "domestic workers",
        "crew member", "crew members", "team member", "team members",
    ]

    stats = {
        "male_term_count": 0,
        "female_term_count": 0,
        "neutral_term_count": 0,
        "response_length": 0,
    }

    for text in texts:
        # Add sentinel spaces to better match term boundaries similar to space-bounded counting
        t = (" " + (text or "").lower() + " ")
        stats["response_length"] += len((text or "").split())

        for term in male_terms:
            stats["male_term_count"] += t.count(" " + term + " ")

        for term in female_terms:
            stats["female_term_count"] += t.count(" " + term + " ")

        for term in neutral_terms:
            stats["neutral_term_count"] += t.count(" " + term + " ")

    num_texts = len(texts)
    if num_texts > 0:
        stats["avg_male_terms"] = stats["male_term_count"] / num_texts
        stats["avg_female_terms"] = stats["female_term_count"] / num_texts
        stats["avg_neutral_terms"] = stats["neutral_term_count"] / num_texts
        stats["avg_response_length"] = stats["response_length"] / num_texts

        total_words = stats["response_length"]
        if total_words > 0:
            stats["male_ratio"] = stats["male_term_count"] / total_words
            stats["female_ratio"] = stats["female_term_count"] / total_words
            stats["neutral_ratio"] = stats["neutral_term_count"] / total_words
        else:
            stats["male_ratio"] = 0.0
            stats["female_ratio"] = 0.0
            stats["neutral_ratio"] = 0.0

    return stats


def analyze_single_text(text: str) -> Dict[str, float]:
    # Wrap single text for reuse of aggregate function
    return analyze_gender_bias([text or ""])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline and masked gender-term proportions for per-row JSONL outputs")
    parser.add_argument("--results_dir", type=str, default="", help="Directory containing per_row*.jsonl files (searches recursively)")
    parser.add_argument("--results_jsonl", type=str, default="", help="Optional single per_row*.jsonl file to include")
    parser.add_argument("--write_per_row_jsonl", type=str, default="", help="Optional path to write per-row evaluation JSONL (aggregated across all inputs)")
    parser.add_argument("--target_genders", type=str, default="",
                        help="Comma-separated list of target genders to include (e.g., 'masculine,feminine,gender-neutral'). If empty, includes all.")
    return parser.parse_args()


def main():
    args = parse_args()

    files: List[str] = []
    if args.results_dir:
        if not os.path.isdir(args.results_dir):
            raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
        for root, _, fnames in os.walk(args.results_dir):
            for fn in fnames:
                if fn.endswith('.jsonl'):
                    files.append(os.path.join(root, fn))
    if args.results_jsonl:
        if not os.path.isfile(args.results_jsonl):
            raise FileNotFoundError(f"Results JSONL not found: {args.results_jsonl}")
        files.append(args.results_jsonl)

    # Deduplicate while preserving order
    seen = set()
    unique_files: List[str] = []
    for fp in files:
        if fp not in seen:
            unique_files.append(fp)
            seen.add(fp)

    if not unique_files:
        raise ValueError("Provide --results_dir (containing .jsonl files) or --results_jsonl.")

    # Aggregation containers across all files
    genders_all: set = set()
    baseline_texts: Dict[str, List[str]] = defaultdict(list)
    # key: (mask_kind, mask_factor, keep_gender, target_gender) -> list of texts
    masked_groups: Dict[Tuple[str, float, str, str], List[str]] = defaultdict(list)

    # Optional per-row output
    write_per_row = bool(args.write_per_row_jsonl)
    per_row_out = None
    if write_per_row:
        per_row_out = open(args.write_per_row_jsonl, "w", encoding="utf-8")

    for fp in unique_files:
        records = load_jsonl(fp)
        if not records:
            continue
        # Attempt to read metadata from first record
        rec0 = records[0]
        rec_genders = rec0.get("genders") or ["male", "female", "neutral"]
        genders_all.update(rec_genders)
        mk = rec0.get("mask_kind", "exclusive")
        try:
            mf = float(rec0.get("mask_factor", 1.0))
        except Exception:
            mf = 1.0

        for rec in records:
            baseline = rec.get("baseline", {}) or {}
            masked_keep = rec.get("masked_keep", {}) or {}

            # Aggregate baseline texts by target gender
            for tg, txt in baseline.items():
                if txt:
                    baseline_texts[tg].append(txt)

            # Aggregate masked texts by group
            for kg, by_tg in masked_keep.items():
                for tg, txt in (by_tg or {}).items():
                    if txt:
                        masked_groups[(mk, mf, kg, tg)].append(txt)

            # Per-row evaluation output
            if write_per_row:
                row_index = rec.get("row_index")
                input_gender = rec.get("input_gender", "")
                per_target = {}
                for tg in rec_genders:
                    base_stats = analyze_single_text(baseline.get(tg, ""))
                    masked_stats = {}
                    for kg in rec_genders:
                        masked_stats[kg] = analyze_single_text((masked_keep.get(kg, {}) or {}).get(tg, ""))
                    per_target[tg] = {
                        "baseline": base_stats,
                        "masked_keep": masked_stats,
                    }

                out_rec = {
                    "row_index": row_index,
                    "input_gender": input_gender,
                    "per_target": per_target,
                    "mask_kind": mk,
                    "mask_factor": mf,
                    "source_file": fp,
                }
                per_row_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    if per_row_out is not None:
        per_row_out.close()
        print(f"Per-row evaluation written to {args.write_per_row_jsonl}")

    # Apply target_genders filter if specified
    if args.target_genders:
        filter_genders = set(g.strip().lower() for g in args.target_genders.split(",") if g.strip())
        genders_all = genders_all & filter_genders

    genders = preferred_gender_order(list(genders_all))
    if not genders:
        # Fallback to old scheme if nothing matches
        preferred_old = ["male", "female", "neutral"]
        genders = preferred_old

    # Aggregate analysis: baseline across all files
    print("\n===== BASELINE (UNMASKED) RESPONSES — AGGREGATED =====")
    baseline_stats: Dict[str, Dict[str, float]] = {}
    baseline_counts: Dict[str, int] = {}
    for tg in genders:
        texts = baseline_texts.get(tg, [])
        baseline_stats[tg] = analyze_gender_bias(texts)
        baseline_counts[tg] = len(texts)
        stats = baseline_stats[tg]
        print(f"\nTARGET={tg.upper()}:")
        print(f"  Number of responses: {baseline_counts[tg]}")
        print(f"  Average response length: {stats.get('avg_response_length', 0):.2f} words")
        print(f"  Male terms: {stats.get('avg_male_terms', 0):.2f} per response ({stats.get('male_ratio', 0):.2%})")
        print(f"  Female terms: {stats.get('avg_female_terms', 0):.2f} per response ({stats.get('female_ratio', 0):.2%})")
        print(f"  Neutral terms: {stats.get('avg_neutral_terms', 0):.2f} per response ({stats.get('neutral_ratio', 0):.2%})")

    # Aggregate analysis: masked groups across all files
    from collections import defaultdict as _dd
    grouped = _dd(list)  # key: (mask_kind, mask_factor) -> list of (kg, tg, stats, count)
    for (mk, mf, kg, tg), texts in masked_groups.items():
        # Filter to only include genders in our target list
        if kg not in genders or tg not in genders:
            continue
        stats = analyze_gender_bias(texts)
        count = len(texts)
        grouped[(mk, mf)].append((kg, tg, stats, count))

    if not grouped:
        print("\nNo masked stats computed from results.")
    else:
        for (mk, mf), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"\n===== MASKED RESPONSES — kind={mk}, strength={mf} (AGGREGATED) =====")
            items.sort(key=lambda x: (x[0], x[1]))
            for kg, tg, stats, count in items:
                print(f"\nKEEP ONLY {kg.upper()} NEURONS on TARGET={tg.upper()}:")
                print(f"  Number of responses: {count}")
                print(f"  Average response length: {stats.get('avg_response_length', 0):.2f} words")
                print(f"  Male terms: {stats.get('avg_male_terms', 0):.2f} per response ({stats.get('male_ratio', 0):.2%})")
                print(f"  Female terms: {stats.get('avg_female_terms', 0):.2f} per response ({stats.get('female_ratio', 0):.2%})")
                print(f"  Neutral terms: {stats.get('avg_neutral_terms', 0):.2f} per response ({stats.get('neutral_ratio', 0):.2%})")

                base = baseline_stats.get(tg, {})
                # Ratio changes
                male_ratio_change = stats.get('male_ratio', 0) - base.get('male_ratio', 0)
                female_ratio_change = stats.get('female_ratio', 0) - base.get('female_ratio', 0)
                neutral_ratio_change = stats.get('neutral_ratio', 0) - base.get('neutral_ratio', 0)
                # Absolute count changes (per response)
                male_abs_change = stats.get('avg_male_terms', 0) - base.get('avg_male_terms', 0)
                female_abs_change = stats.get('avg_female_terms', 0) - base.get('avg_female_terms', 0)
                neutral_abs_change = stats.get('avg_neutral_terms', 0) - base.get('avg_neutral_terms', 0)
                length_change = stats.get('avg_response_length', 0) - base.get('avg_response_length', 0)
                
                print("  CHANGE FROM BASELINE (Ratio):")
                print(f"    Male ratio change: {male_ratio_change:+.2%}")
                print(f"    Female ratio change: {female_ratio_change:+.2%}")
                print(f"    Neutral ratio change: {neutral_ratio_change:+.2%}")
                print("  CHANGE FROM BASELINE (Absolute per response):")
                print(f"    Male terms change: {male_abs_change:+.2f}")
                print(f"    Female terms change: {female_abs_change:+.2f}")
                print(f"    Neutral terms change: {neutral_abs_change:+.2f}")
                print(f"    Response length change: {length_change:+.2f} words")


if __name__ == "__main__":
    main()
